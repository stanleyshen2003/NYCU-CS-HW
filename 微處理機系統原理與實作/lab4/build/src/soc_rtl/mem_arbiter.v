// =============================================================================
//  Program : mem_arbiter.v
//  Author  : Yi-De Lee
//  Date    : Apr/9/2022
// -----------------------------------------------------------------------------
//  Description:
//      The arbiter for mig native interface.
// -----------------------------------------------------------------------------
//  Notes for UG586 7 Series FPGAs Memory Interface Solutions User Guide
//      - app_rdy      : indicates mig user interface is ready to accept command (if app_rdy change from 1 to 0, it means command is accepted)
//      - app_addr     : every address represent 8 bytes, there is only 1GB DDR in KC705 so the address space is 0x07FFFFFF ~ 0x00000000
//      - app_cmd      : 3'b000 for write command, 3'b001 for read command
//      - app_en       : request strobe to mig, app_addr and app_cmd need to be ready
//      - app_wdf_rdy  : indicates that the write data FIFO is ready to receive data
//      - app_wdf_mask : mask for wdf_data, 0 indicates overwrite byte, 1 indicates keep byte
//      - app_wdf_data : data need to be written into ddr
//      - app_wdf_wren : indicates that data on app_wdf_data is valid
//      - app_wdf_end  : indicates that current clock cycle is the last cycle of input data on wdf_data
// -----------------------------------------------------------------------------
//  Revision information:
//
//  Sep/6/2022, by Chun-Jen Tsai:
//    Modify the code to turn constants into parameters so that we can support
//    both the KC705 with CLSIZE = 256 and the Arty boards with CL_SIZE = 128.
//
//    The DRAM parameters on the Arty A7-35T and A7-100T boards:
//    128-bit memory bus, 16-bit word length, 8-beat burst, and 333 MHz memory clock.
//    The DRAM parameters on the Xilinx KC-705 boards:
//    512-bit memory bus, 64-bit word length, 8-beat burst, and 800 MHz memory clock.
// -----------------------------------------------------------------------------
//  License information:
//
//  This software is released under the BSD-3-Clause Licence,
//  see https://opensource.org/licenses/BSD-3-Clause for details.
//  In the following license statements, "software" refers to the
//  "source code" of the complete hardware/software system.
//
//  Copyright 2019,
//                    Embedded Intelligent Systems Lab (EISL)
//                    Deparment of Computer Science
//                    National Chiao Tung Uniersity
//                    Hsinchu, Taiwan.
//
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
// =============================================================================
`include "aquila_config.vh"

module mem_arbiter #(
    parameter XLEN       = 32,
    parameter CLSIZE     = `CLP,
    parameter DRAM_WIDTH = `DRAMP,
    parameter WDF_DBITS  = `WDFP )
(
    // System signals
    input                      clk_i, 
    input                      rst_i,

    // Aquila I-MEM slave interface (read-only)
    input                      S_IMEM_strobe_i,
    input  [XLEN-1 : 0]        S_IMEM_addr_i,
    output                     S_IMEM_done_o,
    output [CLSIZE-1 : 0]      S_IMEM_data_o,

    // Aquila D-MEM slave interface
    input                      S_DMEM_strobe_i,
    input  [XLEN-1 : 0]        S_DMEM_addr_i,
    input                      S_DMEM_rw_i,
    input  [CLSIZE-1 : 0]      S_DMEM_data_i,
    output                     S_DMEM_done_o,
    output [CLSIZE-1 : 0]      S_DMEM_data_o,

    // 7 Series FPGA MIG native user interface
    output [ 27 : 0]           M_MEM_addr_o,
    output [  2 : 0]           M_MEM_cmd_o,
    output                     M_MEM_en_o,
    output [WDF_DBITS-1 : 0]   M_MEM_wdf_data_o,
    output                     M_MEM_wdf_end_o,
    output [WDF_DBITS/8-1 : 0] M_MEM_wdf_mask_o,
    output                     M_MEM_wdf_wren_o,
    input  [WDF_DBITS-1 : 0]   M_MEM_rd_data_i,
    input                      M_MEM_rd_data_valid_i,
    input                      M_MEM_rdy_i,
    input                      M_MEM_wdf_rdy_i,
    output                     M_MEM_sr_req_o,
    output                     M_MEM_ref_req_o,
    output                     M_MEM_zq_req_o
);

    // Strobe source, have two strobe sources (I-MEM, D-MEM) 
    localparam I_STROBE = 0,
               D_STROBE = 1;

    localparam M_IDLE = 0, // wait for strobe
               M_SEND = 1, // wait for mig ready for request
               M_WAIT = 2, // wait for rd data readed from ddr
               M_DONE = 3; // delay one cycle, 'cause we need to reorder
                           // the readed data (for timing improvement)

    localparam DRAM_ADDR_LSBIT = $clog2(DRAM_WIDTH/8);

    // Keep the strobe
    reg                      S_IMEM_strobe_r;
    reg  [XLEN-1 : 0]        S_IMEM_addr_r;

    reg                      S_DMEM_strobe_r;
    reg  [XLEN-1 : 0]        S_DMEM_addr_r;
    reg                      S_DMEM_rw_r;
    reg  [CLSIZE-1 : 0]      S_DMEM_data_r;

    // input selection signals
    reg  [  1 : 0]           sel;
    reg  [ 26 : 0]           addr;
    reg                      rw;
    reg  [WDF_DBITS-1 : 0]   wdata;
    reg  [WDF_DBITS/8-1 : 0] wmask;

    // registered input signals
    reg  [  1 : 0]           sel_r;
    reg  [ 26 : 0]           addr_r;
    reg                      rw_r;
    reg  [WDF_DBITS-1 : 0]   wdata_r;
    reg  [WDF_DBITS/8-1 : 0] wmask_r;

    // FSM signals
    reg  [  1 : 0]           c_state;
    reg  [  1 : 0]           n_state;
    wire                     have_strobe;
    wire                     mig_ready;
    wire [DRAM_WIDTH-1 : 0]  mig_data [0 : 7];

    // data readed from mig need reorder
    reg  [WDF_DBITS-1 : 0]   rdata_reorder;

    //=======================================================
    //  Keep the strobe (in case we miss strobe)
    //=======================================================
    // I-MEM slave interface (read only)
    always @(posedge clk_i) begin
        if (rst_i)
            S_IMEM_strobe_r <= 0;
        else if (S_IMEM_strobe_i)
            S_IMEM_strobe_r <= 1;
        else if (S_IMEM_done_o)
            S_IMEM_strobe_r <= 0; // Clear the strobe
    end

    always @(posedge clk_i) begin
        if (rst_i)
            S_IMEM_addr_r <= 0;
        else if (S_IMEM_strobe_i)
            S_IMEM_addr_r <= S_IMEM_addr_i;
    end

    // D-MEM slave interface
    always @(posedge clk_i) begin
        if (rst_i)
            S_DMEM_strobe_r <= 0;
        else if (S_DMEM_strobe_i)
            S_DMEM_strobe_r <= 1;
        else if (S_DMEM_done_o)
            S_DMEM_strobe_r <= 0;
    end

    always @(posedge clk_i) begin
        if (rst_i)
            S_DMEM_addr_r <= 0;
        else if (S_DMEM_strobe_i)
            S_DMEM_addr_r <= S_DMEM_addr_i;
    end

    always @(posedge clk_i) begin
        if (rst_i)
            S_DMEM_rw_r <= 0;
        else if (S_DMEM_strobe_i)
            S_DMEM_rw_r <= S_DMEM_rw_i;
    end

    always @(posedge clk_i) begin
        if (rst_i)
            S_DMEM_data_r <= 0;
        else if (S_DMEM_strobe_i)
            S_DMEM_data_r <= S_DMEM_data_i;
    end

    //=======================================================
    //  Strobe signals selection (Priority : I-MEM > D-MEM)
    //=======================================================
    always @(*) begin
        if (S_IMEM_strobe_r) 
            sel = I_STROBE;
        else if (S_DMEM_strobe_r) 
            sel = D_STROBE;
        else
            sel = 0;
    end

    always @(*) begin
        case (sel)
            I_STROBE: addr = S_IMEM_addr_r[DRAM_ADDR_LSBIT +: 27];
            D_STROBE: addr = S_DMEM_addr_r[DRAM_ADDR_LSBIT +: 27];
            default : addr = 0;
        endcase
    end

    always @(*) begin
        case (sel)
            I_STROBE: rw = 0; // I-MEM is read-only
            D_STROBE: rw = S_DMEM_rw_r;
            default : rw = 0;
        endcase
    end

    always @(*) begin
        case (sel) // (for scalability, you may assign wdata only with D-MEM's case)
            I_STROBE: wdata = 0; // I-MEM is read-only
`ifdef ARTY
            D_STROBE: wdata = S_DMEM_data_r;
            default : wdata = {CLSIZE{1'b0}};
`else // KC705
            D_STROBE: wdata = (addr[2])? {S_DMEM_data_r, {CLSIZE{1'b0}}} : {{CLSIZE{1'b0}}, S_DMEM_data_r};
            default : wdata = {{CLSIZE{1'b0}}, {CLSIZE{1'b0}}};
`endif
        endcase
    end
    
    always @(*) begin
        case (sel)
`ifdef ARTY
            I_STROBE: wmask = 16'h0000;
            D_STROBE: wmask = 16'h0000;
            default : wmask = 16'h0000;
`else //KC705
            I_STROBE: wmask = (addr[2])? 64'h0000_0000_FFFF_FFFF : 64'hFFFF_FFFF_0000_0000;
            D_STROBE: wmask = (addr[2])? 64'h0000_0000_FFFF_FFFF : 64'hFFFF_FFFF_0000_0000;
            default : wmask = 64'hFFFF_FFFF_FFFF_FFFF;
`endif
        endcase
    end

    //============================================================================
    //  Register selected strobe's signals in IDLE state (for timing improvement)
    //============================================================================
    always @(posedge clk_i) begin
        if (rst_i)
            sel_r <= 0;
        else if (c_state == M_IDLE) 
            sel_r <= sel;
    end

    always @(posedge clk_i) begin
        if (rst_i)
            addr_r <= 0;
        else if (c_state == M_IDLE)
            addr_r <= addr; 
    end

    always @(posedge clk_i) begin
        if (rst_i)
            rw_r <= 0;
        else if (c_state == M_IDLE)
            rw_r <= rw;
    end

    always @(posedge clk_i) begin
        if (rst_i)
            wdata_r <= 0;
        else if (c_state == M_IDLE)
            wdata_r <= wdata;
    end

    always @(posedge clk_i) begin
        if (rst_i)
            wmask_r <= 0;
        else if (c_state == M_IDLE)
            wmask_r <= wmask;
    end

    //=======================================================
    //  Main FSM
    //=======================================================
    always @(posedge clk_i) begin
        if (rst_i)
            c_state <= M_IDLE;
        else
            c_state <= n_state;
    end

    always @(*) begin
        case (c_state)
            M_IDLE: 
                if (have_strobe) 
                    n_state = M_SEND;
                else 
                    n_state = M_IDLE;
            M_SEND: 
                if (mig_ready) 
                    n_state = (rw_r)? M_IDLE : M_WAIT;
                else 
                    n_state = M_SEND;
            M_WAIT:
                if (M_MEM_rd_data_valid_i)
                    n_state = M_DONE; // Delay the output one cycle for reorder read data
                else 
                    n_state = M_WAIT;
            M_DONE:
                n_state = M_IDLE;
        endcase
    end

    assign have_strobe = S_IMEM_strobe_r | S_DMEM_strobe_r;
    assign mig_ready   = (rw_r)? M_MEM_rdy_i && M_MEM_wdf_rdy_i : M_MEM_rdy_i;

    //=======================================================
    //  Logic for MIG native interface
    //=======================================================
    assign mig_data[0] = M_MEM_rd_data_i[(DRAM_WIDTH-1 + DRAM_WIDTH * 0) -: DRAM_WIDTH];
    assign mig_data[1] = M_MEM_rd_data_i[(DRAM_WIDTH-1 + DRAM_WIDTH * 1) -: DRAM_WIDTH];
    assign mig_data[2] = M_MEM_rd_data_i[(DRAM_WIDTH-1 + DRAM_WIDTH * 2) -: DRAM_WIDTH];
    assign mig_data[3] = M_MEM_rd_data_i[(DRAM_WIDTH-1 + DRAM_WIDTH * 3) -: DRAM_WIDTH];
    assign mig_data[4] = M_MEM_rd_data_i[(DRAM_WIDTH-1 + DRAM_WIDTH * 4) -: DRAM_WIDTH];
    assign mig_data[5] = M_MEM_rd_data_i[(DRAM_WIDTH-1 + DRAM_WIDTH * 5) -: DRAM_WIDTH];
    assign mig_data[6] = M_MEM_rd_data_i[(DRAM_WIDTH-1 + DRAM_WIDTH * 6) -: DRAM_WIDTH];
    assign mig_data[7] = M_MEM_rd_data_i[(DRAM_WIDTH-1 + DRAM_WIDTH * 7) -: DRAM_WIDTH];

    // DDR3 read data reorder
    always @(posedge clk_i) begin
        if (rst_i)
            rdata_reorder <= {WDF_DBITS{1'b0}};
        else if (M_MEM_rd_data_valid_i) begin
            case(addr_r[2:0])
                0: rdata_reorder <= {mig_data[7], mig_data[6], mig_data[5], mig_data[4], mig_data[3], mig_data[2], mig_data[1], mig_data[0]};
                1: rdata_reorder <= {mig_data[6], mig_data[5], mig_data[4], mig_data[7], mig_data[2], mig_data[1], mig_data[0], mig_data[3]};
                2: rdata_reorder <= {mig_data[5], mig_data[4], mig_data[7], mig_data[6], mig_data[1], mig_data[0], mig_data[3], mig_data[2]};
                3: rdata_reorder <= {mig_data[4], mig_data[7], mig_data[6], mig_data[5], mig_data[0], mig_data[3], mig_data[2], mig_data[1]};
                4: rdata_reorder <= {mig_data[3], mig_data[2], mig_data[1], mig_data[0], mig_data[7], mig_data[6], mig_data[5], mig_data[4]};
                5: rdata_reorder <= {mig_data[2], mig_data[1], mig_data[0], mig_data[3], mig_data[6], mig_data[5], mig_data[4], mig_data[7]};       
                6: rdata_reorder <= {mig_data[1], mig_data[0], mig_data[3], mig_data[2], mig_data[5], mig_data[4], mig_data[7], mig_data[6]};
                7: rdata_reorder <= {mig_data[0], mig_data[3], mig_data[2], mig_data[1], mig_data[4], mig_data[7], mig_data[6], mig_data[5]};
            endcase
        end
    end

    //=======================================================
    //  Output logic
    //=======================================================
`ifdef ARTY
    assign S_IMEM_data_o = rdata_reorder;
    assign S_DMEM_data_o = rdata_reorder;
`else
    assign S_IMEM_data_o = (addr_r[2])? rdata_reorder[511:256] : rdata_reorder[255:0];
    assign S_DMEM_data_o = (addr_r[2])? rdata_reorder[511:256] : rdata_reorder[255:0];
`endif
    assign S_NMEM_data_o = rdata_reorder;
    assign S_IMEM_done_o = (c_state != M_IDLE) && (n_state == M_IDLE) && (sel_r == I_STROBE);
    assign S_DMEM_done_o = (c_state != M_IDLE) && (n_state == M_IDLE) && (sel_r == D_STROBE);

    assign M_MEM_addr_o     = {1'b0, addr_r}; // the 28th bit always 0, because we only have one bank
    assign M_MEM_cmd_o      = (rw_r)? 3'b000 : 3'b001;
    assign M_MEM_en_o       = (c_state == M_SEND) && mig_ready;
    assign M_MEM_wdf_data_o = wdata_r;
    assign M_MEM_wdf_wren_o = (c_state == M_SEND) && mig_ready && rw_r;
    assign M_MEM_wdf_end_o  = (c_state == M_SEND) && mig_ready && rw_r;
    assign M_MEM_wdf_mask_o = wmask_r;
    assign M_MEM_sr_req_o   = 0;
    assign M_MEM_ref_req_o  = 0;
    assign M_MEM_zq_req_o   = 0;

endmodule
