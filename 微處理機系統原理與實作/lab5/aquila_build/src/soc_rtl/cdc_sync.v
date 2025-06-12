// =============================================================================
//  Program : cdc_sync.v
//  Author  : Po-wei Ho
//  Date    : Sep/14/2020
// -----------------------------------------------------------------------------
//  Description:
//  This is the Clock domain crossing synchronizer of the Aquila core (A RISC-V core).
//
//  Every signal has two FSM to control write and read operations of asynchronous FIFO.
//
//  After all signals are ready, this unit will output all synchronized signals for 
//  only one clock cycle.
// -----------------------------------------------------------------------------
//  Revision information:
//
//  Aug/02/2024, by Chun-Jen Tsai:
//    Fix the Clock-Domain Crossing issue. The original code did not use two reset
//    signals, one for each domain, to properly reset signals in each domain.
//    Another bug is that it uses DMEM_rw_r, instead of DMEM_rw_i in the clk_core
//    clock domain.
//
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

module cdc_sync #( parameter XLEN = 32, parameter CLSIZE = `CLP )
(
    // System signals
    input                 clk_core,
    input                 clk_memc,
    input                 rst_core_i,
    input                 rst_memc_i,

    // Aquila ICACHE port interface signals
    input                 IMEM_strobe_i,
    input  [XLEN-1 : 0]   IMEM_addr_i,
    output                IMEM_done_o,
    output [CLSIZE-1 : 0] IMEM_data_o,

    // Aquila DCACHE port interface signals
    input                 DMEM_strobe_i,
    input  [XLEN-1 : 0]   DMEM_addr_i,
    input                 DMEM_rw_i,
    input  [CLSIZE-1 : 0] DMEM_wt_data_i,
    output                DMEM_done_o,
    output [CLSIZE-1 : 0] DMEM_rd_data_o,

    // MIG ICACHE port interface signals
    output                IMEM_strobe_o,
    output [XLEN-1 : 0]   IMEM_addr_o,
    input                 IMEM_done_i,
    input  [CLSIZE-1 : 0] IMEM_data_i,

    // MIG DCACHE port interface signals
    output                DMEM_strobe_o,
    output [XLEN-1 : 0]   DMEM_addr_o,
    output                DMEM_rw_o,
    output [CLSIZE-1 : 0] DMEM_wt_data_o,
    input                 DMEM_done_i,
    input  [CLSIZE-1 : 0] DMEM_rd_data_i
);

// FSM States
localparam
    IMEM_memc_in_IDLE = 0,
    IMEM_memc_in_WAIT = 1;
reg A, A_nxt;

localparam
    DMEM_memc_in_IDLE = 0,
    DMEM_memc_in_WAIT = 1;
reg B, B_nxt;

localparam
    IMEM_core_out_IDLE = 0,
    IMEM_core_out_WAIT = 1;
reg C, C_nxt;

localparam
    DMEM_core_out_IDLE = 0,
    DMEM_core_out_WAIT = 1;
reg D, D_nxt;

localparam
    IMEM_done_core_out_IDLE = 0,
    IMEM_done_core_out_WAIT = 1;
reg E, E_nxt;

localparam
    DMEM_done_core_out_IDLE = 0,
    DMEM_done_core_out_WAIT = 1;
reg F, F_nxt;

localparam
    DMEM_rd_data_core_out_IDLE = 0,
    DMEM_rd_data_core_out_WAIT = 1;
reg G, G_nxt;

localparam
    IMEM_rd_data_core_out_IDLE = 0,
    IMEM_rd_data_core_out_WAIT = 1;
reg H, H_nxt;

localparam
    IMEM_memc_out_IDLE = 0,
    IMEM_memc_out_WAIT = 1;
reg Q, Q_nxt;

localparam
    IMEM_strobe_memc_out_IDLE = 0,
    IMEM_strobe_memc_out_WAIT = 1;
reg R, R_nxt;

localparam
    IMEM_addr_memc_out_IDLE = 0,
    IMEM_addr_memc_out_WAIT = 1;
reg S, S_nxt;

localparam
    DMEM_memc_out_IDLE = 0,
    DMEM_memc_out_WAIT = 1;
reg T, T_nxt;

localparam
    DMEM_strobe_memc_out_IDLE = 0,
    DMEM_strobe_memc_out_WAIT = 1;
reg U, U_nxt;

localparam
    DMEM_addr_memc_out_IDLE = 0,
    DMEM_addr_memc_out_WAIT = 1;
reg V, V_nxt;

localparam
    DMEM_rw_memc_out_IDLE = 0,
    DMEM_rw_memc_out_WAIT = 1;
reg W, W_nxt;

localparam
    DMEM_wt_data_memc_out_IDLE = 0,
    DMEM_wt_data_memc_out_WAIT = 1;
reg X, X_nxt;

// registers for fifo out
reg                 IMEM_strobe_r;
reg [XLEN-1 : 0]    IMEM_addr_r;
reg                 DMEM_strobe_r;
reg [XLEN-1 : 0]    DMEM_addr_r;
reg                 DMEM_rw_r;
reg [CLSIZE-1 : 0]  DMEM_wt_data_r;

reg                 IMEM_done_r;
reg                 DMEM_done_r;
reg [CLSIZE-1 : 0]  IMEM_rd_data_r;
reg [CLSIZE-1 : 0]  DMEM_rd_data_r;

wire DMEM_memc_out_ready;
assign DMEM_memc_out_ready = T == DMEM_memc_out_IDLE && T_nxt == DMEM_memc_out_WAIT;
wire IMEM_memc_out_ready;
assign IMEM_memc_out_ready = Q == IMEM_memc_out_IDLE && Q_nxt == IMEM_memc_out_WAIT;
wire IMEM_core_out_ready;
assign IMEM_core_out_ready = C == IMEM_core_out_IDLE && C_nxt == IMEM_core_out_WAIT;
wire DMEM_core_out_ready;
assign DMEM_core_out_ready = D == DMEM_core_out_IDLE && D_nxt == DMEM_core_out_WAIT;

// output to mig
assign IMEM_strobe_o  = IMEM_memc_out_ready? IMEM_strobe_r : 0;
assign IMEM_addr_o    = IMEM_memc_out_ready? IMEM_addr_r : 0;
assign DMEM_strobe_o  = DMEM_memc_out_ready? DMEM_strobe_r : 0;
assign DMEM_addr_o    = DMEM_memc_out_ready? DMEM_addr_r : 0;
assign DMEM_rw_o      = DMEM_memc_out_ready? DMEM_rw_r : 0;
assign DMEM_wt_data_o = DMEM_memc_out_ready? DMEM_wt_data_r : 0;

// output to core
assign IMEM_done_o    = IMEM_core_out_ready? IMEM_done_r : 0;
assign DMEM_done_o    = DMEM_core_out_ready? DMEM_done_r : 0;
assign IMEM_data_o    = IMEM_core_out_ready? IMEM_rd_data_r : 0;
assign DMEM_rd_data_o = DMEM_core_out_ready? DMEM_rd_data_r : 0;

//=======================================================
//  Async_fifo signals
//=======================================================
// from core to mig
wire IMEM_strobe_full, IMEM_addr_full;
wire IMEM_strobe_empty, IMEM_addr_empty;

wire DMEM_strobe_full, DMEM_addr_full, DMEM_rw_full, DMEM_wt_data_full;
wire DMEM_strobe_empty, DMEM_addr_empty, DMEM_rw_empty, DMEM_wt_data_empty;

wire IMEM_strobe_out;
wire [XLEN-1 : 0] IMEM_addr_out;
wire DMEM_strobe_out;
wire [XLEN-1 : 0] DMEM_addr_out;
wire DMEM_rw_out;
wire [CLSIZE-1 : 0] DMEM_wt_data_out;

// form mig to core
wire IMEM_done_full, DMEM_done_full, IMEM_rd_data_full, DMEM_rd_data_full;
wire IMEM_done_empty, DMEM_done_empty, IMEM_rd_data_empty, DMEM_rd_data_empty;

wire IMEM_done_out;
wire DMEM_done_out;
wire [CLSIZE-1 : 0] IMEM_rd_data_out;
wire [CLSIZE-1 : 0] DMEM_rd_data_out;

//=======================================================
//  Mig input registers
//=======================================================
reg IMEM_done_i_r, DMEM_done_i_r;
reg [CLSIZE-1 : 0] IMEM_rd_data_i_r, DMEM_rd_data_i_r;

always @(posedge clk_memc)
begin
    if (rst_memc_i) begin
       IMEM_done_i_r    <= 0; 
       IMEM_rd_data_i_r <= 0;
    end        
    else if (IMEM_done_i) begin
        IMEM_done_i_r    <= IMEM_done_i;
        IMEM_rd_data_i_r <= IMEM_data_i;
    end        
    else if (IMEM_strobe_o) begin
        IMEM_done_i_r    <= 0;  
        IMEM_rd_data_i_r <= 0;      
    end

end

always @(posedge clk_memc)
begin
    if (rst_memc_i) begin
        DMEM_done_i_r    <= 0;
        DMEM_rd_data_i_r <= 0;
    end        
    else if (DMEM_done_i) begin
        DMEM_done_i_r    <= DMEM_done_i;
        DMEM_rd_data_i_r <= DMEM_rd_data_i;
    end
    else if (DMEM_strobe_o) begin
        DMEM_done_i_r    <= 0;
        DMEM_rd_data_i_r <= 0;
    end
end


/* *******************************************************************************
 * Finite State Machines from cpu to fifo                                         *
 * *******************************************************************************/
//=======================================================
//  FSM IMEM of Core clock domain input
//=======================================================
localparam
    IMEM_core_clk_in_IDLE = 0,
    IMEM_core_clk_in_WAIT = 1;

reg O, O_nxt;

wire IMEM_in_full_all, IMEM_in_wr_en;
assign IMEM_in_full_all = IMEM_strobe_full & IMEM_addr_full;
assign IMEM_in_wr_en = (O == IMEM_core_clk_in_IDLE && O_nxt == IMEM_core_clk_in_WAIT);

always @(posedge clk_core)
begin
    if (rst_core_i)
        O <= IMEM_core_clk_in_IDLE;
    else
        O <= O_nxt;
end

always @(*)
begin
    case (O)
        IMEM_core_clk_in_IDLE: O_nxt = (IMEM_strobe_i & !IMEM_in_full_all)? IMEM_core_clk_in_WAIT : IMEM_core_clk_in_IDLE;
        IMEM_core_clk_in_WAIT: O_nxt = (IMEM_strobe_i)? IMEM_core_clk_in_WAIT : IMEM_core_clk_in_IDLE;
    endcase
end

//=======================================================
//  FSM DMEM of Core clock domain input
//=======================================================
localparam
    DMEM_core_clk_in_IDLE = 0,
    DMEM_core_clk_in_WAIT = 1;

reg P, P_nxt;

wire DMEM_in_full_all, DMEM_in_wr_en;
assign DMEM_in_full_all = DMEM_strobe_full & DMEM_addr_full & DMEM_rw_full & DMEM_wt_data_full;
assign DMEM_in_wr_en = (P == DMEM_core_clk_in_IDLE && P_nxt == DMEM_core_clk_in_WAIT);

always @(posedge clk_core)
begin
    if (rst_core_i)
        P <= DMEM_core_clk_in_IDLE;
    else
        P <= P_nxt;
end

always @(*)
begin
    case (P)
        DMEM_core_clk_in_IDLE: P_nxt = (DMEM_strobe_i & !DMEM_in_full_all)? DMEM_core_clk_in_WAIT : DMEM_core_clk_in_IDLE;
        DMEM_core_clk_in_WAIT: P_nxt = (DMEM_strobe_i)? DMEM_core_clk_in_WAIT : DMEM_core_clk_in_IDLE;
    endcase
end

/* *******************************************************************************
 * FSM from fifo to mig                                         *
 * *******************************************************************************/
//=======================================================
//  FSM IMEM of MEMC clock domain output
//=======================================================
always @(posedge clk_memc)
begin
    if (rst_memc_i)
        Q <= IMEM_memc_out_IDLE;
    else
        Q <= Q_nxt;
end

always @(*)
begin
    case (Q)
        IMEM_memc_out_IDLE: Q_nxt = (R == IMEM_strobe_memc_out_WAIT && S == IMEM_addr_memc_out_WAIT)? IMEM_memc_out_WAIT : IMEM_memc_out_IDLE;
        IMEM_memc_out_WAIT: Q_nxt = IMEM_memc_out_IDLE;
    endcase
end

//=======================================================
//  FSM IMEM_strobe of MEMC clock domain output
//=======================================================
always @(posedge clk_memc)
begin
    if (rst_memc_i)
        R <= IMEM_strobe_memc_out_IDLE;
    else
        R <= R_nxt;
end

always @(*)
begin
    case (R)
        IMEM_strobe_memc_out_IDLE: R_nxt = (!IMEM_strobe_empty)? IMEM_strobe_memc_out_WAIT : IMEM_strobe_memc_out_IDLE;
        IMEM_strobe_memc_out_WAIT: R_nxt = (IMEM_memc_out_ready)? IMEM_strobe_memc_out_IDLE : IMEM_strobe_memc_out_WAIT;
    endcase
end

always @(posedge clk_memc)
begin
    if (rst_memc_i)
        IMEM_strobe_r <= 0;
    else if (!IMEM_strobe_empty)
        IMEM_strobe_r <= IMEM_strobe_out;
end

//=======================================================
//  FSM IMEM_addr of MEMC clock domain output
//=======================================================
always @(posedge clk_memc)
begin
    if (rst_memc_i)
        S <= IMEM_addr_memc_out_IDLE;
    else
        S <= S_nxt;
end

always @(*)
begin
    case (S)
        IMEM_addr_memc_out_IDLE: S_nxt = (!IMEM_addr_empty)? IMEM_addr_memc_out_WAIT : IMEM_addr_memc_out_IDLE;
        IMEM_addr_memc_out_WAIT: S_nxt = (IMEM_memc_out_ready)? IMEM_addr_memc_out_IDLE : IMEM_addr_memc_out_WAIT;
    endcase
end

always @(posedge clk_memc)
begin
    if (rst_memc_i)
        IMEM_addr_r <= 0;
    else if (!IMEM_addr_empty)
        IMEM_addr_r <= IMEM_addr_out;
end

//=======================================================
//  FSM DMEM of MEMC clock domain output
//=======================================================
always @(posedge clk_memc)
begin
    if (rst_memc_i)
        T <= DMEM_memc_out_IDLE;
    else
        T <= T_nxt;
end

always @(*)
begin
    case (T)
        DMEM_memc_out_IDLE: T_nxt = (U == DMEM_strobe_memc_out_WAIT && V == DMEM_addr_memc_out_WAIT && W == DMEM_rw_memc_out_WAIT && X == DMEM_wt_data_memc_out_WAIT)? DMEM_memc_out_WAIT : DMEM_memc_out_IDLE;
        DMEM_memc_out_WAIT: T_nxt = DMEM_memc_out_IDLE;
    endcase
end

//=======================================================
//  FSM DMEM_strobe of MEMC clock domain output
//=======================================================
always @(posedge clk_memc)
begin
    if (rst_memc_i)
        U <= DMEM_strobe_memc_out_IDLE;
    else
        U <= U_nxt;
end

always @(*)
begin
    case (U)
        DMEM_strobe_memc_out_IDLE: U_nxt = (!DMEM_strobe_empty)? DMEM_strobe_memc_out_WAIT : DMEM_strobe_memc_out_IDLE;
        DMEM_strobe_memc_out_WAIT: U_nxt = (DMEM_memc_out_ready)? DMEM_strobe_memc_out_IDLE : DMEM_strobe_memc_out_WAIT;
    endcase
end

always @(posedge clk_memc)
begin
    if (rst_memc_i)
        DMEM_strobe_r <= 0;
    else if (!DMEM_strobe_empty)
        DMEM_strobe_r <= DMEM_strobe_out;
end

//=======================================================
//  FSM DMEM_addr of MEMC clock domain output
//=======================================================
always @(posedge clk_memc)
begin
    if (rst_memc_i)
        V <= DMEM_addr_memc_out_IDLE;
    else
        V <= V_nxt;
end

always @(*)
begin
    case (V)
        DMEM_addr_memc_out_IDLE: V_nxt = (!DMEM_addr_empty)? DMEM_addr_memc_out_WAIT : DMEM_addr_memc_out_IDLE;
        DMEM_addr_memc_out_WAIT: V_nxt = (DMEM_memc_out_ready)? DMEM_addr_memc_out_IDLE : DMEM_addr_memc_out_WAIT;
    endcase
end

always @(posedge clk_memc)
begin
    if (rst_memc_i)
        DMEM_addr_r <= 0;
    else if (!DMEM_addr_empty)
        DMEM_addr_r <= DMEM_addr_out;
end

//=======================================================
//  FSM DMEM_rw of MEMC clock domain output
//=======================================================
always @(posedge clk_memc)
begin
    if (rst_memc_i)
        W <= DMEM_rw_memc_out_IDLE;
    else
        W <= W_nxt;
end

always @(*)
begin
    case (W)
        DMEM_rw_memc_out_IDLE: W_nxt = (!DMEM_rw_empty)? DMEM_rw_memc_out_WAIT : DMEM_rw_memc_out_IDLE;
        DMEM_rw_memc_out_WAIT: W_nxt = (DMEM_memc_out_ready)? DMEM_rw_memc_out_IDLE : DMEM_rw_memc_out_WAIT;
    endcase
end

always @(posedge clk_memc)
begin
    if (rst_memc_i)
        DMEM_rw_r <= 0;
    else if (!DMEM_rw_empty)
        DMEM_rw_r <= DMEM_rw_out;
end

//=======================================================
//  FSM DMEM_wt_data of MEMC clock domain output
//=======================================================
always @(posedge clk_memc)
begin
    if (rst_memc_i)
        X <= DMEM_wt_data_memc_out_IDLE;
    else
        X <= X_nxt;
end

always @(*)
begin
    case (X)
        DMEM_wt_data_memc_out_IDLE: X_nxt = (!DMEM_wt_data_empty)? DMEM_wt_data_memc_out_WAIT : DMEM_wt_data_memc_out_IDLE;
        DMEM_wt_data_memc_out_WAIT: X_nxt = (DMEM_memc_out_ready)? DMEM_wt_data_memc_out_IDLE : DMEM_wt_data_memc_out_WAIT;
    endcase
end

always @(posedge clk_memc)
begin
    if (rst_memc_i)
        DMEM_wt_data_r <= 0;
    else if (!DMEM_wt_data_empty)
        DMEM_wt_data_r <= DMEM_wt_data_out;
end

/* *******************************************************************************
 * FSM from mig to fifo                                         *
 * *******************************************************************************/
//=======================================================
//  FSM IMEM of MEMC clock domain input
//=======================================================
wire IMEM_done_in_full_all, IMEM_done_in_wr_en;
assign IMEM_done_in_full_all = IMEM_done_full & IMEM_rd_data_full;
assign IMEM_done_in_wr_en = (A == IMEM_memc_in_IDLE && A_nxt == IMEM_memc_in_WAIT);

always @(posedge clk_memc)
begin
    if (rst_memc_i)
        A <= IMEM_memc_in_IDLE;
    else
        A <= A_nxt;
end

always @(*)
begin
    case (A)
        IMEM_memc_in_IDLE: A_nxt = (IMEM_done_i_r & !IMEM_done_in_full_all)? IMEM_memc_in_WAIT : IMEM_memc_in_IDLE;
        IMEM_memc_in_WAIT: A_nxt = (IMEM_done_i_r)? IMEM_memc_in_WAIT : IMEM_memc_in_IDLE;
    endcase
end

//=======================================================
//  FSM DMEM of MEMC clock domain input
//=======================================================
wire DMEM_done_in_wr_en;
assign DMEM_done_in_wr_en = (B == DMEM_memc_in_IDLE && B_nxt == DMEM_memc_in_WAIT);

always @(posedge clk_memc)
begin
    if (rst_memc_i)
        B <= DMEM_memc_in_IDLE;
    else
        B <= B_nxt;
end


always @(*)
begin
    case (B)
        DMEM_memc_in_IDLE: 
            if (DMEM_done_i_r & !DMEM_done_full) begin
                if (!DMEM_rw_r) begin //read means need to wait rdata
                    if (!DMEM_rd_data_full) begin
                        B_nxt = DMEM_memc_in_WAIT;
                    end 
                    else begin
                        B_nxt = DMEM_memc_in_IDLE;
                    end
                end 
                else begin
                        B_nxt = DMEM_memc_in_WAIT;
                end
            end 
            else begin
                B_nxt = DMEM_memc_in_IDLE;
            end
            
        DMEM_memc_in_WAIT: B_nxt = (DMEM_done_i_r)? DMEM_memc_in_WAIT : DMEM_memc_in_IDLE;
    endcase
end

/* *******************************************************************************
 * Finite State Machines from fifo to cpu                                        *
 * *******************************************************************************/
//=======================================================
//  FSM IMEM of Core clock domain output
//=======================================================
always @(posedge clk_core)
begin
    if (rst_core_i)
        C <= IMEM_core_out_IDLE;
    else
        C <= C_nxt;
end


always @(*)
begin
    case (C)
        IMEM_core_out_IDLE: C_nxt = (E == IMEM_done_core_out_WAIT && H == IMEM_rd_data_core_out_WAIT)? IMEM_core_out_WAIT : IMEM_core_out_IDLE;
        IMEM_core_out_WAIT: C_nxt = IMEM_core_out_IDLE;
    endcase
end

//=======================================================
//  FSM DMEM of Core clock domain output
//=======================================================
always @(posedge clk_core)
begin
    if (rst_core_i)
        D <= DMEM_core_out_IDLE;
    else
        D <= D_nxt;
end


always @(*)
begin
    case (D)
        DMEM_core_out_IDLE: 
            if (F == DMEM_done_core_out_WAIT) begin
                if (!DMEM_rw_i) begin //read means need to wait rdata
                    if (G == DMEM_rd_data_core_out_WAIT) begin
                        D_nxt = DMEM_core_out_WAIT;
                    end 
                    else begin
                        D_nxt = DMEM_core_out_IDLE;
                    end
                end 
                else begin
                    D_nxt = DMEM_core_out_WAIT;
                end
            end 
            else begin
                D_nxt = DMEM_core_out_IDLE;
            end

        DMEM_core_out_WAIT: D_nxt = DMEM_core_out_IDLE;
    endcase
end

//=======================================================
//  FSM IMEM_done of Core clock domain output
//=======================================================
always @(posedge clk_core)
begin
    if (rst_core_i)
        E <= IMEM_done_core_out_IDLE;
    else
        E <= E_nxt;
end

always @(*)
begin
    case (E)
        IMEM_done_core_out_IDLE: E_nxt = (!IMEM_done_empty)? IMEM_done_core_out_WAIT : IMEM_done_core_out_IDLE;
        IMEM_done_core_out_WAIT: E_nxt = (IMEM_core_out_ready)? IMEM_done_core_out_IDLE : IMEM_done_core_out_WAIT;
    endcase
end

always @(posedge clk_core)
begin
    if (rst_core_i)
        IMEM_done_r <= 0;
    else if (!IMEM_done_empty)
        IMEM_done_r <= IMEM_done_out;
end

//=======================================================
//  FSM DMEM_done of Core clock domain output
//=======================================================
always @(posedge clk_core)
begin
    if (rst_core_i)
        F <= DMEM_done_core_out_IDLE;
    else
        F <= F_nxt;
end

always @(*)
begin
    case (F)
        DMEM_done_core_out_IDLE: F_nxt = (!DMEM_done_empty)? DMEM_done_core_out_WAIT : DMEM_done_core_out_IDLE;
        DMEM_done_core_out_WAIT: F_nxt = (DMEM_core_out_ready)? DMEM_done_core_out_IDLE : DMEM_done_core_out_WAIT;
    endcase
end

always @(posedge clk_core)
begin
    if (rst_core_i)
        DMEM_done_r <= 0;
    else if (!DMEM_done_empty)
        DMEM_done_r <= DMEM_done_out;
end

//=======================================================
//  FSM IMEM_rd_data of Core clock domain output
//=======================================================
always @(posedge clk_core)
begin
    if (rst_core_i)
        H <= IMEM_rd_data_core_out_IDLE;
    else
        H <= H_nxt;
end

always @(*)
begin
    case (H)
        IMEM_rd_data_core_out_IDLE: H_nxt = (!IMEM_rd_data_empty)? IMEM_rd_data_core_out_WAIT : IMEM_rd_data_core_out_IDLE;
        IMEM_rd_data_core_out_WAIT: H_nxt = (IMEM_core_out_ready)? IMEM_rd_data_core_out_IDLE : IMEM_rd_data_core_out_WAIT;
    endcase
end

always @(posedge clk_core)
begin
    if (rst_core_i)
        IMEM_rd_data_r <= 0;
    else if (!IMEM_rd_data_empty)
        IMEM_rd_data_r <= IMEM_rd_data_out;
end

//=======================================================
//  FSM DMEM_rd_data of Core clock domain output
//=======================================================
always @(posedge clk_core)
begin
    if (rst_core_i)
        G <= DMEM_rd_data_core_out_IDLE;
    else
        G <= G_nxt;
end

always @(*)
begin
    case (G)
        DMEM_rd_data_core_out_IDLE: G_nxt = (!DMEM_rd_data_empty)? DMEM_rd_data_core_out_WAIT : DMEM_rd_data_core_out_IDLE;
        DMEM_rd_data_core_out_WAIT: G_nxt = (DMEM_core_out_ready)? DMEM_rd_data_core_out_IDLE : DMEM_rd_data_core_out_WAIT;
    endcase
end

always @(posedge clk_core)
begin
    if (rst_core_i)
        DMEM_rd_data_r <= 0;
    else if (!DMEM_rd_data_empty)
        DMEM_rd_data_r <= DMEM_rd_data_out;
end


/* *******************************************************************************
 * Async FIFO mofules                                                            *
 * *******************************************************************************/
async_fifo_signal IMEM_strobe
(
    .full(IMEM_strobe_full),
    .din(IMEM_strobe_i),
    .wr_en(IMEM_in_wr_en),
    .empty(IMEM_strobe_empty),
    .dout(IMEM_strobe_out),
    .rd_en(!IMEM_strobe_empty),
    .wr_clk(clk_core),
    .rd_clk(clk_memc)
);

async_fifo_addr IMEM_addr
(
    .full(IMEM_addr_full),
    .din(IMEM_addr_i),
    .wr_en(IMEM_in_wr_en),
    .empty(IMEM_addr_empty),
    .dout(IMEM_addr_out),
    .rd_en(!IMEM_addr_empty),
    .wr_clk(clk_core),
    .rd_clk(clk_memc)
);

async_fifo_signal DMEM_strobe
(
    .full(DMEM_strobe_full),
    .din(DMEM_strobe_i),
    .wr_en(DMEM_in_wr_en),
    .empty(DMEM_strobe_empty),
    .dout(DMEM_strobe_out),
    .rd_en(!DMEM_strobe_empty),
    .wr_clk(clk_core),
    .rd_clk(clk_memc)
);

async_fifo_signal DMEM_rw
(
    .full(DMEM_rw_full),
    .din(DMEM_rw_i),
    .wr_en(DMEM_in_wr_en),
    .empty(DMEM_rw_empty),
    .dout(DMEM_rw_out),
    .rd_en(!DMEM_rw_empty),
    .wr_clk(clk_core),
    .rd_clk(clk_memc)
);

async_fifo_addr DMEM_addr
(
    .full(DMEM_addr_full),
    .din(DMEM_addr_i),
    .wr_en(DMEM_in_wr_en),
    .empty(DMEM_addr_empty),
    .dout(DMEM_addr_out),
    .rd_en(!DMEM_addr_empty),
    .wr_clk(clk_core),
    .rd_clk(clk_memc)
);

async_fifo_data DMEM_wt_data
(
    .full(DMEM_wt_data_full),
    .din(DMEM_wt_data_i),
    .wr_en(DMEM_in_wr_en),
    .empty(DMEM_wt_data_empty),
    .dout(DMEM_wt_data_out),
    .rd_en(!DMEM_wt_data_empty),
    .wr_clk(clk_core),
    .rd_clk(clk_memc)
);

async_fifo_signal IMEM_done
(
    .full(IMEM_done_full),
    .din(IMEM_done_i_r),
    .wr_en(IMEM_done_in_wr_en),
    .empty(IMEM_done_empty),
    .dout(IMEM_done_out),
    .rd_en(!IMEM_done_empty),
    .wr_clk(clk_memc),
    .rd_clk(clk_core)
);

async_fifo_data IMEM_rd_data
(
    .full(IMEM_rd_data_full),
    .din(IMEM_rd_data_i_r),
    .wr_en(IMEM_done_in_wr_en),
    .empty(IMEM_rd_data_empty),
    .dout(IMEM_rd_data_out),
    .rd_en(!IMEM_rd_data_empty),
    .wr_clk(clk_memc),
    .rd_clk(clk_core)
);

async_fifo_signal DMEM_done
(
    .full(DMEM_done_full),
    .din(DMEM_done_i_r),
    .wr_en(DMEM_done_in_wr_en),
    .empty(DMEM_done_empty),
    .dout(DMEM_done_out),
    .rd_en(!DMEM_done_empty),
    .wr_clk(clk_memc),
    .rd_clk(clk_core)
);

async_fifo_data DMEM_rd_data
(
    .full(DMEM_rd_data_full),
    .din(DMEM_rd_data_i_r),
    .wr_en(!DMEM_rw_r & DMEM_done_in_wr_en),
    .empty(DMEM_rd_data_empty),
    .dout(DMEM_rd_data_out),
    .rd_en(!DMEM_rd_data_empty),
    .wr_clk(clk_memc),
    .rd_clk(clk_core)
);
endmodule
