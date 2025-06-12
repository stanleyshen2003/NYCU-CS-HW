`timescale 1ns / 1ps
// =============================================================================
//  Program : mig_7series_sim.v
//  Author  : Chun-Jen Tsai
//  Date    : Sep/03/2023
// -----------------------------------------------------------------------------
//  Description:
//  This module is for Memory controller + DRAM simulation.  Note that this DRAM
//  simulatoion model does not really simulate true DRAM behavior.
//
//  For example, for efficiency reasons, the true DRAM may reorder the read
//  data based on the least significant three address bit of the request address.
//  However, since we only simulate DRAM for the cache controller and the cache
//  controller only requests memory reads with addresses which align to 8-byte
//  boundaries, therefore, no data reordering is simulated here.
//
//  We also assume that there is no delay to burst write since most memory controller
//  has write registers that stores write in one cycle.  The read delay is
//  parameterized.
// -----------------------------------------------------------------------------
//  Revision information:
//
//  None.
// -----------------------------------------------------------------------------
//  License information:
//
//  This software is released under the BSD-3-Clause Licence,
//  see https://opensource.org/licenses/BSD-3-Clause for details.
//  In the following license statements, "software" refers to the
//  "source code" of the complete hardware/software system.
//
//  Copyright 2023,
//                    Embedded Intelligent Systems Lab (EISL)
//                    Deparment of Computer Science
//                    National Yang Ming Chiao Tung Uniersity (NYCU)
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

module mig_7series_sim
#(parameter DATA_WIDTH = `WDFP,
            N_ENTRIES = 65536,
            ADDRW = $clog2(N_ENTRIES),
            DRAM_BURST_DELAY = 12)
(
    // memory controller clock & reset
    input clk_i,
    input rst_i,

    // user-logic interface ports
    input                    app_en,
    input    [ 27 : 0]       app_addr,
    input    [  2 : 0]       app_cmd,
    input    [ `WDFP-1 : 0]  app_wdf_data,
    input                    app_wdf_end,
    input    [`WDFP/8-1 : 0] app_wdf_mask,
    input                    app_wdf_wren,
    output reg [`WDFP-1 : 0] app_rd_data,
    output reg               app_rd_data_end,
    output reg               app_rd_data_valid,
    output                   app_rdy,
    output                   app_wdf_rdy,
    input                    app_sr_req,
    input                    app_ref_req,
    input                    app_zq_req,
    output                   app_sr_active,
    output                   app_ref_ack,
    output                   app_zq_ack
);

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

reg [DATA_WIDTH-1 : 0] DRAM [N_ENTRIES-1 : 0];
reg [`DRAMP-1 : 0]     mig_data [0 : 7];
reg [7:0]              delay_counter = 0;
reg                    delay_trig, delay_trig_ff;

assign app_rdy = ~(| delay_counter); // Simulated MIG/DRAM is ready for commands.
assign app_wdf_rdy = 1'b1;           // Simulated MIG/DRAM is ready for writing.

assign app_sr_active = 1'b0; // Dummy response.
assign app_ref_ack = 1'b1;   // Dummy response.
assign app_zq_ack = 1'b1;    // Dummy response.

// ---------------
// Read operation
// ---------------
always@(posedge clk_i)
begin
    delay_trig <= app_en;
    delay_trig_ff <= delay_trig;
end

always@(posedge clk_i)
begin
    if (delay_trig == 1 && delay_trig_ff == 0)
        delay_counter <= DRAM_BURST_DELAY;
    else
        delay_counter <=  delay_counter - (| delay_counter);
end

always@(posedge clk_i)
begin
    if (delay_counter == 1)
    begin
        mig_data[0] <= DRAM[app_addr][`DRAMP*0 +: `DRAMP];
        mig_data[1] <= DRAM[app_addr][`DRAMP*1 +: `DRAMP];
        mig_data[2] <= DRAM[app_addr][`DRAMP*2 +: `DRAMP];
        mig_data[3] <= DRAM[app_addr][`DRAMP*3 +: `DRAMP];
        mig_data[4] <= DRAM[app_addr][`DRAMP*4 +: `DRAMP];
        mig_data[5] <= DRAM[app_addr][`DRAMP*5 +: `DRAMP];
        mig_data[6] <= DRAM[app_addr][`DRAMP*6 +: `DRAMP];
        mig_data[7] <= DRAM[app_addr][`DRAMP*7 +: `DRAMP];
        app_rd_data_end <= 1;
        app_rd_data_valid <= 1;
    end
    else
    begin
        app_rd_data_end <= 0;
        app_rd_data_valid <= 0;
    end
end

// DDR3 read data reorder, must match the reorder process in the DRAM spec.
always @(*) begin
    if (app_rdy) begin
        case(app_addr[2:0])
            0: app_rd_data <= {mig_data[7], mig_data[6], mig_data[5], mig_data[4], mig_data[3], mig_data[2], mig_data[1], mig_data[0]};
            1: app_rd_data <= {mig_data[4], mig_data[7], mig_data[6], mig_data[5], mig_data[0], mig_data[3], mig_data[2], mig_data[1]};
            2: app_rd_data <= {mig_data[5], mig_data[4], mig_data[7], mig_data[6], mig_data[1], mig_data[0], mig_data[3], mig_data[2]};
            3: app_rd_data <= {mig_data[6], mig_data[5], mig_data[4], mig_data[7], mig_data[2], mig_data[1], mig_data[0], mig_data[3]};
            4: app_rd_data <= {mig_data[3], mig_data[2], mig_data[1], mig_data[0], mig_data[7], mig_data[6], mig_data[5], mig_data[4]};
            5: app_rd_data <= {mig_data[0], mig_data[3], mig_data[2], mig_data[1], mig_data[4], mig_data[7], mig_data[6], mig_data[5]};       
            6: app_rd_data <= {mig_data[1], mig_data[0], mig_data[3], mig_data[2], mig_data[5], mig_data[4], mig_data[7], mig_data[6]};
            7: app_rd_data <= {mig_data[2], mig_data[1], mig_data[0], mig_data[3], mig_data[6], mig_data[5], mig_data[4], mig_data[7]};
        endcase
    end
end

// ----------------
// Write operation
// ----------------
integer idx;

always@(posedge clk_i)
begin
    if (app_en && !app_cmd[0])
    begin
        for (idx = 0; idx < DATA_WIDTH/8; idx = idx + 1)
            if (!app_wdf_mask[idx])
                DRAM[app_addr][(idx<<3) +: 8] <= app_wdf_data[(idx<<3) +: 8];
    end
end
endmodule   // dram_sim
