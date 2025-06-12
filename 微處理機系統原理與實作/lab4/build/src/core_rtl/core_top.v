    `timescale 1ns / 1ps
// =============================================================================
//  Program : core_top.v
//  Author  : Jin-you Wu
//  Date    : Dec/19/2018
// -----------------------------------------------------------------------------
//  Description:
//  This is the top-level Aquila IP wrapper for an AXI-based application
//  processor SoC.
//
//  The pipeline architecture of Aquila 1.0 was based on the Microblaze-
//  compatible processor, KernelBlaze, originally designed by Dong-Fong Syu.
//  This file, core_top.v, was derived from CPU.v of KernelBlaze by Dong-Fong
//  on Sep/09/2017.
//
// -----------------------------------------------------------------------------
//  Revision information:
//
//  Oct/16-17/2019, by Chun-Jen Tsai:
//    Unified the memory accesses scheme of the processor core, pushing the
//    address decoding of different memory devices to the SoC level.  Change
//    the initial program counter address from a "PARAMETER' to an input
//    signal, which comes from a system register at the SoC-level.
//
//  Nov/29/2019, by Chun-Jen Tsai:
//    Change the overall pipeline architecture of Aquila. Merges the pipeline
//    register moduels of Fetch, Decode, and Execute stages into the respective
//    moudules.
//
//  Aug/06/2020, by Jen-Yu Chi:
//    modify irq_taken and the pc that is written to mepc.
//
//  Aug/15/2020, by Chun-Jen Tsai:
//    Removed the Unconditional Branch Prediction Unit and merged its function
//    into the BPU.
//
//  Sep/16/2022, by Chun-Jen Tsai:
//    Disable interrupts during external memory accesses or AMO operations.
//
//  Aug/20/2024, by Chun-Jen Tsai:
//    Modify the Memory and Writeback stages such that it matches the coding
//    style of the other stages. Also, changes the naming convention of the
//    inter-stage signals to make them more readable.
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

module core_top #(
    parameter HART_ID       = 0,
    parameter XLEN          = 32
)
(
    // System signals.
    input                 clk_i,
    input                 rst_i,
    input                 stall_i,

    // Program counter address at reset.
    input  [XLEN-1 : 0]   init_pc_addr_i,

    // Instruction memory port.
    input  [XLEN-1 : 0]   code_i,
    input                 code_ready_i,
    output [XLEN-1 : 0]   code_addr_o,
    output                code_req_o,

    // Data or device memory port.
    input  [XLEN-1 : 0]   data_i,
    input                 data_ready_i,
    output [XLEN-1 : 0]   data_o,
    output [XLEN-1 : 0]   data_addr_o,
    output                data_rw_o,      // 0: data read, 1: data write.
    output [XLEN/8-1 : 0] data_byte_enable_o,
    output                data_req_o,
    output                data_is_amo_o,
    output [ 4: 0]        data_amo_type_o,
    input                 data_addr_ext_i,

    // Cache flush signal.
    output                cache_flush_o,

    // Interrupt sources.
    input                 ext_irq_i,
    input                 tmr_irq_i,
    input                 sft_irq_i
);

// ------------------------------
//  Fetch stage output signals
// ------------------------------
wire [XLEN-1 : 0] fet2dec_instr;
wire [XLEN-1 : 0] fet2dec_pc;
wire              fet_branch_hit;
wire              fet_branch_decision;

wire              fet2dec_valid;
wire              fet2dec_xcpt_valid;
wire [ 3 : 0]     fet2dec_xcpt_cause;
wire [XLEN-1 : 0] fet2dec_xcpt_tval;

// ------------------------------
//  Decode stage output signals
// ------------------------------
// Signals sent to multiple destinations
wire [XLEN-1 : 0] dec_pc;
wire              dec_is_branch;
wire              dec_is_jal;
wire              dec_is_jalr;
wire              dec_branch_hit;
wire              dec_branch_decision;

// Signals sent to Pipeline Control
wire              dec2plc_load_hazard;
wire              dec_unsupported_instr;

// Signals sent to Register File
wire [ 4 : 0]     dec2rfu_rs1_addr;
wire [ 4 : 0]     dec2rfu_rs2_addr;

// Signals sent to Forwarding Unit
wire [ 4 : 0]     dec2fwd_rs1_addr;
wire [ 4 : 0]     dec2fwd_rs2_addr;
wire [XLEN-1 : 0] dec2fwd_rs1_data;
wire [XLEN-1 : 0] dec2fwd_rs2_data;
wire [11: 0]      dec2fwd_csr_addr;
wire [XLEN-1 : 0] dec2fwd_csr_data;

// Signals sent to CSR
wire [11 : 0]     dec2csr_csr_addr;

// Signals sent to Execute
wire [ 4 : 0]     dec2exe_rd_addr;
wire              dec2exe_we;
wire              dec2exe_re;
wire [XLEN-1 : 0] dec2exe_imm;
wire              dec2exe_csr_we;
wire [ 4 : 0]     dec2exe_csr_imm;
wire [ 1 : 0]     dec2exe_inA_sel;
wire [ 1 : 0]     dec2exe_inB_sel;
wire [ 1 : 0]     dec2exe_dsize_sel;
wire [ 2 : 0]     dec2exe_regfile_sel;
wire [ 2 : 0]     dec2exe_operation_sel;
wire              dec2exe_regfile_we;
wire              dec2exe_signex_sel;
wire              dec2exe_alu_muldiv_sel;
wire              dec2exe_shift_sel;
wire              dec2exe_is_fencei;
wire              dec2exe_is_amo;
wire [ 4 : 0]     dec2exe_amo_type;

wire              dec2exe_fetch_valid;
wire              dec2exe_sys_jump;
wire [ 1 : 0]     dec2exe_sys_jump_csr_addr;
wire              dec2exe_xcpt_valid;
wire [ 3 : 0]     dec2exe_xcpt_cause;
wire [XLEN-1 : 0] dec2exe_xcpt_tval;

// ------------------------------
//  Execute stage output signals
// ------------------------------
wire              exe_branch_taken;
wire [XLEN-1 : 0] exe_branch_restore_pc;
wire [XLEN-1 : 0] exe_branch_target_addr;
wire              exe_is_branch2bpu;
wire              exe_re;
wire              exe_we;

wire              exe_branch_misprediction;
wire              exe_is_fencei;
wire              exe_is_amo2mem;
wire [ 4 : 0]     exe_amo_type2mem;

// to FWD unit and Memory stage
wire [XLEN-1 : 0] exe_p_data;
wire              exe_csr_we;
wire [11 : 0]     exe_csr_addr;
wire [XLEN-1 : 0] exe_csr_data;

// to Memory stage
wire              exe2mem_regfile_we;
wire              exe2mem_signex_sel;
wire [ 4 : 0]     exe2mem_rd_addr;
wire [ 2 : 0]     exe2mem_regfile_input_sel;
wire [ 1 : 0]     exe2mem_dsize_sel;
wire [XLEN-1 : 0] exe2mem_rs2_data;
wire [XLEN-1 : 0] exe2mem_addr;

wire              exe2mem_fetch_valid;
wire              exe2mem_sys_jump;
wire [ 1 : 0]     exe2mem_sys_jump_csr_addr;
wire              exe2mem_xcpt_valid;
wire [ 3 : 0]     exe2mem_xcpt_cause;
wire [XLEN-1 : 0] exe2mem_xcpt_tval;
wire [XLEN-1 : 0] exe2mem_pc;

// ------------------------------
//  Memory stage output signals
// ------------------------------
wire [XLEN-1 : 0] mem_dataout;
wire [ 3 : 0]     mem_byte_sel;
wire              mem_align_exception;

wire              mem2wbk_regfile_we;
wire              mem2wbk_signex_sel;
wire [ 4 : 0]     mem2wbk_rd_addr;
wire [ 2 : 0]     mem2wbk_regfile_input_sel;
wire [XLEN-1 : 0] mem2wbk_aligned_data;
wire [XLEN-1 : 0] mem2wbk_p_data;

wire              mem2wbk_fetch_valid;
wire              mem2wbk_sys_jump;
wire [ 1 : 0]     mem2wbk_sys_jump_csr_addr;
wire              mem2wbk_xcpt_valid;
wire [ 3 : 0]     mem2wbk_xcpt_cause;
wire [XLEN-1 : 0] mem2wbk_xcpt_tval;
wire [XLEN-1 : 0] mem2wbk_pc;
wire              mem2wbk_csr_we;
wire [11 : 0]     mem2wbk_csr_addr;
wire [XLEN-1 : 0] mem2wbk_csr_data;

// --------------------------------
//  Writeback stage output signals
// --------------------------------
wire [XLEN-1 : 0] wbk_rd_data;
wire              wbk_rd_we;
wire [ 4 : 0]     wbk_rd_addr;

wire              wbk_csr_we;
wire [11 : 0]     wbk_csr_addr;
wire [XLEN-1 : 0] wbk_csr_data;

wire              wbk2csr_fetch_valid;
wire              wbk2csr_sys_jump;
wire [ 1 : 0]     wbk2csr_sys_jump_csr_addr;
wire              wbk2csr_xcpt_valid;
wire [ 3 : 0]     wbk2csr_xcpt_cause;
wire [XLEN-1 : 0] wbk2csr_xcpt_tval;
wire [XLEN-1 : 0] wbk2csr_pc;

// ---------------------------------
//  Output signals from other units
// ---------------------------------
// PipeLine Control (PLC) unit
wire plc2fet_flush;
wire plc2dec_flush;
wire plc2exe_flush;
wire plc2mem_flush;
wire plc2wbk_flush;

// Program Counter Unit (PCU)
wire [XLEN-1 : 0] pcu_pc;

// ForWarding Unit (FWD)
wire [XLEN-1 : 0] fwd_rs1;
wire [XLEN-1 : 0] fwd_rs2;
wire [XLEN-1 : 0] fwd2exe_csr_data;

// Register File (RFU)
wire [XLEN-1 : 0] rfu2dec_rs1_data;
wire [XLEN-1 : 0] rfu2dec_rs2_data;

// Control Status Registers (CSR)
wire              csr_irq_taken;
reg               csr_irq_taken_r;
wire [XLEN-1 : 0] csr_pc_handler;
wire [XLEN-1 : 0] csr2dec_data;
wire              csr_sys_jump;
wire [XLEN-1 : 0] csr_sys_jump_data;

// Branch Prediction Unit (BPU)
wire              bpu_branch_hit;
wire              bpu_branch_decision;
wire [XLEN-1 : 0] bpu_branch_target_addr;

// Misc. signals
wire              irq_enable;
wire              irq_taken;
reg  [XLEN-1 : 0] nxt_unwb_PC;
wire [ 1 : 0]     privilege_level;

// =============================================================================
//  Signals sent to the instruction & data memory IPs in the Aquila SoC
//
assign code_addr_o = pcu_pc;
assign data_addr_o = exe2mem_addr;
assign data_rw_o = exe_we;
assign data_o = mem_dataout;
assign data_byte_enable_o = mem_byte_sel;

assign cache_flush_o = exe_is_fencei;

// =============================================================================
//  Atomic operation signals from Execute to Memory
//
assign data_is_amo_o = exe_is_amo2mem;
assign data_amo_type_o = exe_amo_type2mem;

// =============================================================================
//  Control signals to temporarily disable interrupts
//  We must avoid AMO operations or external memory/device
//  accesses being interrupted.
//
assign irq_enable = ~((data_addr_ext_i && (exe_we|exe_re)) || exe_is_amo2mem);

// =============================================================================
// Finite state machine that controls the processor pipeline stalls.
//
localparam i_NEXT = 0, i_WAIT = 1;
localparam d_IDLE = 0, d_WAIT = 1, d_STALL = 2;
reg iS, iS_nxt;
reg [1:0] dS, dS_nxt;

// -----------------------------------------------------------------------------
// The stall signals:
//    # stall_pipeline stalls the entire pipeline stages
//    # stall_data_hazard only stall the Program_Counter and the Fetch stages
//
wire stall_data_hazard; // The stall signal from Pipeline Control.
wire stall_from_exe;    // The stall signal from Execute.
wire stall_instr_fetch;
wire stall_data_fetch;
wire stall_pipeline;

assign stall_instr_fetch = (!code_ready_i);
assign stall_data_fetch = (dS_nxt == d_WAIT) && (! exe_is_fencei);
assign stall_pipeline = stall_instr_fetch | stall_data_fetch | stall_from_exe;

// Maintain irq_taken signal for pipeline stall
assign irq_taken = csr_irq_taken | csr_irq_taken_r;

always @(posedge clk_i)
begin
    if (rst_i)
        csr_irq_taken_r <= 0;
    else if (stall_instr_fetch | stall_data_fetch | stall_from_exe)
        csr_irq_taken_r <= csr_irq_taken_r | csr_irq_taken;
    else
        csr_irq_taken_r <= 0;
end

// =============================================================================
always@(*) begin
    if (!wbk2csr_xcpt_valid) begin
        if (mem2wbk_fetch_valid)
            nxt_unwb_PC = mem2wbk_pc;
        else if (exe2mem_fetch_valid)
            nxt_unwb_PC = exe2mem_pc;
        else if (dec2exe_fetch_valid)
            nxt_unwb_PC = dec_pc;
        else if(fet2dec_valid)
            nxt_unwb_PC = fet2dec_pc;
        else
            nxt_unwb_PC = pcu_pc;
    end else begin
        nxt_unwb_PC = wbk2csr_pc;
    end
end

// =============================================================================
// Finite state machine that controls the instruction & data fetches.
//
always @(posedge clk_i)
begin
    if (rst_i)
        iS <= i_NEXT;
    else
        iS <= iS_nxt;
end

always @(*)
begin
    case (iS)
        i_NEXT: // CJ Tsai 0227_2020: I-fetch when I-memory ready.
            if (code_ready_i)
                iS_nxt = i_NEXT;
            else
                iS_nxt = i_WAIT;
        i_WAIT:
            if (code_ready_i)
                iS_nxt = i_NEXT; // one-cycle delay
            else
                iS_nxt = i_WAIT;
    endcase
end

always @(posedge clk_i)
begin
    if (rst_i)
        dS <= d_IDLE;
    else
        dS <= dS_nxt;
end

always @(*)
begin
    case (dS)
        d_IDLE:
            if ((exe_re || exe_we) && !mem_align_exception)
                dS_nxt = d_WAIT;
            else
                dS_nxt = d_IDLE;
        d_WAIT:
            if (data_ready_i)
                if (stall_instr_fetch || stall_from_exe)
                    dS_nxt = d_STALL;
                else
                    dS_nxt = d_IDLE;
            else
                dS_nxt = d_WAIT;
        d_STALL:
            // CY Hsiang July 20 2020
            if (stall_instr_fetch || stall_from_exe)
                dS_nxt = d_STALL;
            else
                dS_nxt = d_IDLE;
        default:
            dS_nxt = d_IDLE;
    endcase
end

// -----------------------------------------------------------------------------
// Output instruction/data request signals
assign code_req_o = (iS == i_NEXT);
assign data_req_o = (dS == d_IDLE) && (exe_re || exe_we);

// -----------------------------------------------------------------------------
// Data Memory Signals and logic
// CY Hsiang July 20 2020
reg  [XLEN-1 : 0] data_read_reg;
wire [XLEN-1 : 0] data_read2wbk;

always @(posedge clk_i) begin
    if (rst_i)
        data_read_reg <= 0;
    else if (data_ready_i)
        data_read_reg <= data_i;
end

assign data_read2wbk = (dS == d_STALL) ? data_read_reg : data_i;

////////////////////////////////////////////////////////////////////////////////
//                        the following are submodules                        //
////////////////////////////////////////////////////////////////////////////////

// =============================================================================
pipeline_control Pipeline_Control(
    // from Decode
    .unsupported_instr_i(dec_unsupported_instr),
    .branch_hit_i(dec_branch_hit),
    .is_load_hazard(dec2plc_load_hazard),

    // from Execute
    .branch_taken_i(exe_branch_taken),
    .branch_misprediction_i(exe_branch_misprediction),
    .is_fencei_i(exe_is_fencei),

    // System Jump operation
    .sys_jump_i(csr_sys_jump),

    // to Fetch
    .flush2fet_o(plc2fet_flush),

    // to Decode
    .flush2dec_o(plc2dec_flush),

    // to Execute
    .flush2exe_o(plc2exe_flush),

    // to Memory
    .flush2mem_o(plc2mem_flush),

    // to Writeback
    .flush2wbk_o(plc2wbk_flush),

    // to PCU and Fetch
    .data_hazard_o(stall_data_hazard)
);

// =============================================================================
forwarding_unit Forwarding_Unit(
    // from Decode
    .rs1_addr_i(dec2fwd_rs1_addr),
    .rs2_addr_i(dec2fwd_rs2_addr),
    .csr_addr_i(dec2fwd_csr_addr),
    .rs1_data_i(dec2fwd_rs1_data),
    .rs2_data_i(dec2fwd_rs2_data),
    .csr_data_i(dec2fwd_csr_data),

    // from Execute
    .exe_rd_addr_i(exe2mem_rd_addr),
    .exe_regfile_we_i(exe2mem_regfile_we),
    .exe_regfile_input_sel_i(exe2mem_regfile_input_sel),
    .exe_p_data_i(exe_p_data),

    .exe_csr_addr_i(exe_csr_addr),
    .exe_csr_we_i(exe_csr_we),
    .exe_csr_data_i(exe_csr_data),

    // from Writeback
    .wbk_rd_addr_i(wbk_rd_addr),
    .wbk_regfile_we_i(wbk_rd_we),
    .wbk_rd_data_i(wbk_rd_data),

    .wbk_csr_we_i(wbk_csr_we),
    .wbk_csr_addr_i(wbk_csr_addr),
    .wbk_csr_data_i(wbk_csr_data),

    // to Execute and CSR
    .rs1_o(fwd_rs1),
    .rs2_o(fwd_rs2),
    .csr_data_o(fwd2exe_csr_data)
);

// =============================================================================
bpu #(.XLEN(XLEN)) Branch_Prediction_Unit(
    // Top-level system signals
    .clk_i(clk_i),
    .rst_i(rst_i),
    .stall_i(stall_pipeline),

    // from Program_Counter
    .pc_i(pcu_pc),

    // from Decode
    .is_jal_i(dec_is_jal),
    .is_cond_branch_i(dec_is_branch),
    .dec_pc_i(dec_pc),

    // from Execute
    .exe_is_branch_i(exe_is_branch2bpu),
    .branch_taken_i(exe_branch_taken),
    .branch_misprediction_i(exe_branch_misprediction),
    .branch_target_addr_i(exe_branch_target_addr),

    // to Program_Counter and Fetch
    .branch_hit_o(bpu_branch_hit),
    .branch_decision_o(bpu_branch_decision),
    .branch_target_addr_o(bpu_branch_target_addr)
);

// =============================================================================
reg_file Register_File(
    // Top-level system signals
    .clk_i(clk_i),
    .rst_i(rst_i),

    // from Decode
    .rs1_addr_i(dec2rfu_rs1_addr),
    .rs2_addr_i(dec2rfu_rs2_addr),

    // from Writeback
    .rd_we_i(wbk_rd_we),
    .rd_addr_i(wbk_rd_addr),
    .rd_data_i(wbk_rd_data),

    // to Decode
    .rs1_data_o(rfu2dec_rs1_data),
    .rs2_data_o(rfu2dec_rs2_data)
);

// =============================================================================
program_counter Program_Counter(
    // Top-level system signals
    .clk_i(clk_i),
    .rst_i(rst_i),

    // Program Counter address at reset
    .init_pc_addr_i(init_pc_addr_i),

    // Interrupt
    .irq_taken_i(irq_taken),
    .PC_handler_i(csr_pc_handler),

    // Stall signal for Program Counter
    .stall_i(stall_pipeline || (stall_data_hazard && !irq_taken)),

    // from BPU
    .bpu_branch_hit_i(bpu_branch_hit),
    .bpu_branch_decision_i(bpu_branch_decision),
    .bpu_branch_target_addr_i(bpu_branch_target_addr),

    // System Jump operation
    .sys_jump_i(csr_sys_jump),
    .sys_jump_data_i(csr_sys_jump_data),

    // frome Decode
    .dec_branch_hit_i(dec_branch_hit),
    .dec_branch_decision_i(dec_branch_decision),
    .dec_pc_i(dec_pc),

    // from Execute
    .exe_branch_misprediction_i(exe_branch_misprediction),
    .exe_branch_taken_i(exe_branch_taken),
    .exe_branch_target_addr_i(exe_branch_target_addr),
    .exe_branch_restore_addr_i(exe_branch_restore_pc),
    .is_fencei_i(exe_is_fencei),

    // to Fetch, I-memory
    .pc_o(pcu_pc)
);

// =============================================================================
fetch Fetch(
    // Top-level system signals
    .clk_i(clk_i),
    .rst_i(rst_i),
    .stall_i(stall_pipeline || (stall_data_hazard && !irq_taken)),

    // from Pipeline Control and CSR
    .flush_i(plc2fet_flush || irq_taken),

    // from BPU
    .branch_hit_i(bpu_branch_hit),
    .branch_decision_i(bpu_branch_decision),

    // from I-memory
    .instruction_i(code_i),

    // PC of the current instruction.
    .pc_i(pcu_pc),
    .pc_o(fet2dec_pc),

    // to Decode
    .instruction_o(fet2dec_instr),
    .branch_hit_o(fet_branch_hit),
    .branch_decision_o(fet_branch_decision),

     // Has instruction fetch being successiful?
    .fetch_valid_o(fet2dec_valid),   // Validity of the Fetch stage.  
    .xcpt_valid_o(fet2dec_xcpt_valid), // Any valid exception?
    .xcpt_cause_o(fet2dec_xcpt_cause), // Cause of the exception (if any).
    .xcpt_tval_o(fet2dec_xcpt_tval)    // Trap Value (if any).
);

// =============================================================================
decode Decode(
    // Top-level system signals
    .clk_i(clk_i),
    .rst_i(rst_i),
    .stall_i(stall_pipeline),

    // Processor pipeline flush signal.
    .flush_i(plc2dec_flush || irq_taken),

    // Signals from Fetch.
    .instruction_i(fet2dec_instr),
    .branch_hit_i(fet_branch_hit),
    .branch_decision_i(fet_branch_decision),

    // Signals from CSR.
    .csr_data_i(csr2dec_data),
    .privilege_lvl_i(privilege_level),

    // Instruction operands from the Register File. To be forwarded.
    .rs1_data_i(rfu2dec_rs1_data),
    .rs2_data_i(rfu2dec_rs2_data),

    // Operand register IDs to the Register File
    .rs1_addr_o(dec2rfu_rs1_addr),
    .rs2_addr_o(dec2rfu_rs2_addr),

    // illegal instruction
    .unsupported_instr_o(dec_unsupported_instr),

    // to Execute
    .imm_o(dec2exe_imm),
    .csr_we_o(dec2exe_csr_we),
    .csr_imm_o(dec2exe_csr_imm),
    .inputA_sel_o(dec2exe_inA_sel),
    .inputB_sel_o(dec2exe_inB_sel),
    .operation_sel_o(dec2exe_operation_sel),
    .alu_muldiv_sel_o(dec2exe_alu_muldiv_sel),
    .shift_sel_o(dec2exe_shift_sel),
    .branch_hit_o(dec_branch_hit), //also to PLC and PCU
    .branch_decision_o(dec_branch_decision),
    .is_jalr_o(dec_is_jalr),
    .is_fencei_o(dec2exe_is_fencei),

    // to Execute and BPU
    .is_branch_o(dec_is_branch),
    .is_jal_o(dec_is_jal),

    // to CSR
    .csr_addr_o(dec2csr_csr_addr),

    // to Execute
    .rd_addr_o(dec2exe_rd_addr),
    .regfile_we_o(dec2exe_regfile_we),
    .regfile_input_sel_o(dec2exe_regfile_sel),
    .we_o(dec2exe_we),
    .re_o(dec2exe_re),
    .dsize_sel_o(dec2exe_dsize_sel),
    .signex_sel_o(dec2exe_signex_sel),
    .is_amo_o(dec2exe_is_amo),
    .amo_type_o(dec2exe_amo_type),

    // to Pipeline Control
    .is_load_hazard_o(dec2plc_load_hazard),

    // to Forwarding Unit
    .rs1_addr2fwd_o(dec2fwd_rs1_addr),
    .rs2_addr2fwd_o(dec2fwd_rs2_addr),
    .rs1_data2fwd_o(dec2fwd_rs1_data),
    .rs2_data2fwd_o(dec2fwd_rs2_data),

    .csr_addr2fwd_o(dec2fwd_csr_addr), // also to Execute
    .csr_data2fwd_o(dec2fwd_csr_data),

    // PC of the current instruction.
    .pc_i(fet2dec_pc),
    .pc_o(dec_pc),

    // System Jump operation
    .sys_jump_o(dec2exe_sys_jump),
    .sys_jump_csr_addr_o(dec2exe_sys_jump_csr_addr),

    // Has instruction fetch being successiful?
    .fetch_valid_i(fet2dec_valid),
    .fetch_valid_o(dec2exe_fetch_valid),

    // Exception info passed from Fetch to Execute.
    .xcpt_valid_i(fet2dec_xcpt_valid),
    .xcpt_cause_i(fet2dec_xcpt_cause),
    .xcpt_tval_i(fet2dec_xcpt_tval),
    .xcpt_valid_o(dec2exe_xcpt_valid),
    .xcpt_cause_o(dec2exe_xcpt_cause),
    .xcpt_tval_o(dec2exe_xcpt_tval)
);

// =============================================================================
execute Execute(
    // Top-level system signals
    .clk_i(clk_i),
    .rst_i(rst_i),

    // Pipeline stall signal.
    .stall_i(stall_instr_fetch | stall_data_fetch),

    // Processor pipeline flush signal.
    .flush_i(plc2exe_flush || irq_taken),

    // Signals from the Decode stage.
    .imm_i(dec2exe_imm),
    .inputA_sel_i(dec2exe_inA_sel),
    .inputB_sel_i(dec2exe_inB_sel),
    .operation_sel_i(dec2exe_operation_sel),
    .alu_muldiv_sel_i(dec2exe_alu_muldiv_sel),
    .shift_sel_i(dec2exe_shift_sel),
    .is_branch_i(dec_is_branch),
    .is_jal_i(dec_is_jal),
    .is_jalr_i(dec_is_jalr),
    .is_fencei_i(dec2exe_is_fencei),
    .branch_hit_i(dec_branch_hit),
    .branch_decision_i(dec_branch_decision),

    .regfile_we_i(dec2exe_regfile_we),
    .regfile_input_sel_i(dec2exe_regfile_sel),
    .we_i(dec2exe_we),
    .re_i(dec2exe_re),
    .dsize_sel_i(dec2exe_dsize_sel),
    .signex_sel_i(dec2exe_signex_sel),
    .rd_addr_i(dec2exe_rd_addr),
    .is_amo_i(dec2exe_is_amo),
    .amo_type_i(dec2exe_amo_type),

    .csr_imm_i(dec2exe_csr_imm),
    .csr_we_i(dec2exe_csr_we),
    .csr_we_addr_i(dec2fwd_csr_addr),

    // Signals from the Forwarding Unit.
    .rs1_data_i(fwd_rs1),
    .rs2_data_i(fwd_rs2),
    .csr_data_i(fwd2exe_csr_data),

    // Branch prediction singnals to PLC, PCU, and BPU.
    .is_branch_o(exe_is_branch2bpu),
    .branch_taken_o(exe_branch_taken),
    .branch_misprediction_o(exe_branch_misprediction),
    .branch_target_addr_o(exe_branch_target_addr),     // to PCU and BPU
    .branch_restore_pc_o(exe_branch_restore_pc),       // to PCU only

    // Pipeline stall signal generator, activated when executing
    //    multicycle mul, div and rem instructions.
    .stall_from_exe_o(stall_from_exe),

    // Signals to D-Memory.
    .we_o(exe_we),
    .re_o(exe_re),
    .is_fencei_o(exe_is_fencei),
    .is_amo_o(exe_is_amo2mem),
    .amo_type_o(exe_amo_type2mem),

    // Signals to the Memory stage.
    .rs2_data_o(exe2mem_rs2_data),
    .addr_o(exe2mem_addr),
    .dsize_sel_o(exe2mem_dsize_sel),

    // Signals to the Memory, Writeback stages and the Forwarding Units.
    .regfile_we_o(exe2mem_regfile_we),
    .regfile_input_sel_o(exe2mem_regfile_input_sel),
    .rd_addr_o(exe2mem_rd_addr),
    .signex_sel_o(exe2mem_signex_sel),
    .p_data_o(exe_p_data),

    .csr_we_o(exe_csr_we),
    .csr_we_addr_o(exe_csr_addr),
    .csr_we_data_o(exe_csr_data),

    // PC of the current instruction.
    .pc_i(dec_pc),
    .pc_o(exe2mem_pc),

    // System Jump operations
    .sys_jump_i(dec2exe_sys_jump),
    .sys_jump_o(exe2mem_sys_jump),
    .sys_jump_csr_addr_i(dec2exe_sys_jump_csr_addr),
    .sys_jump_csr_addr_o(exe2mem_sys_jump_csr_addr),

    // Has instruction fetch being successiful?
    .fetch_valid_i(dec2exe_fetch_valid),
    .fetch_valid_o(exe2mem_fetch_valid),

    // Exception info passed from Decode to Memory.
    .xcpt_valid_i(dec2exe_xcpt_valid),
    .xcpt_cause_i(dec2exe_xcpt_cause),
    .xcpt_tval_i(dec2exe_xcpt_tval),
    .xcpt_valid_o(exe2mem_xcpt_valid),
    .xcpt_cause_o(exe2mem_xcpt_cause),
    .xcpt_tval_o(exe2mem_xcpt_tval)
);

// =============================================================================
memory Memory(
    // Top-level system signals
    .clk_i(clk_i),
    .rst_i(rst_i),
    .stall_i(stall_pipeline),

    // Writeback stage flush signal.
    .flush_i(plc2mem_flush || irq_taken),

    // from Execute stage
    .mem_addr_i(exe2mem_addr),
    .unaligned_data_i(exe2mem_rs2_data),      // store value
    .dsize_sel_i(exe2mem_dsize_sel),
    .we_i(exe_we),
    .re_i(exe_re),
    .p_data_i(exe_p_data),

    .regfile_we_i(exe2mem_regfile_we),
    .regfile_input_sel_i(exe2mem_regfile_input_sel),
    .rd_addr_i(exe2mem_rd_addr),
    .signex_sel_i(exe2mem_signex_sel),

    .csr_we_i(exe_csr_we),
    .csr_we_addr_i(exe_csr_addr),
    .csr_we_data_i(exe_csr_data),

    // from D-memory
    .m_data_i(data_read2wbk),

    // to D-memory
    .data_o(mem_dataout),                     // data_write
    .byte_sel_o(mem_byte_sel),

    // to Writeback stage
    .regfile_we_o(mem2wbk_regfile_we),
    .regfile_input_sel_o(mem2wbk_regfile_input_sel),
    .rd_addr_o(mem2wbk_rd_addr),
    .signex_sel_o(mem2wbk_signex_sel),
    .aligned_data_o(mem2wbk_aligned_data),
    .p_data_o(mem2wbk_p_data),

    .csr_we_o(mem2wbk_csr_we),
    .csr_we_addr_o(mem2wbk_csr_addr),
    .csr_we_data_o(mem2wbk_csr_data),

    // Exception signal for memory mis-alignment.
    .mem_align_exception_o(mem_align_exception),

    // PC of the current instruction.
    .pc_i(exe2mem_pc),
    .pc_o(mem2wbk_pc),

    // System Jump operations
    .sys_jump_i(exe2mem_sys_jump),
    .sys_jump_o(mem2wbk_sys_jump),
    .sys_jump_csr_addr_i(exe2mem_sys_jump_csr_addr),
    .sys_jump_csr_addr_o(mem2wbk_sys_jump_csr_addr),

    // Has instruction fetch being successiful?
    .fetch_valid_i(exe2mem_fetch_valid),
    .fetch_valid_o(mem2wbk_fetch_valid),

    // Exception info passed from Execute to Writeback.
    .xcpt_valid_i(exe2mem_xcpt_valid),
    .xcpt_cause_i(exe2mem_xcpt_cause),
    .xcpt_tval_i(exe2mem_xcpt_tval),
    .xcpt_valid_o(mem2wbk_xcpt_valid),
    .xcpt_cause_o(mem2wbk_xcpt_cause),
    .xcpt_tval_o(mem2wbk_xcpt_tval)
);

// =============================================================================
writeback Writeback(
    // Top-level system signals
    .clk_i(clk_i),
    .rst_i(rst_i),
    .stall_i(stall_pipeline),

    // Writeback stage flush signal.
    .flush_i(plc2wbk_flush || irq_taken),

    // from Memory stage
    .regfile_we_i(mem2wbk_regfile_we),
    .regfile_input_sel_i(mem2wbk_regfile_input_sel),
    .rd_addr_i(mem2wbk_rd_addr),
    .signex_sel_i(mem2wbk_signex_sel),
    .aligned_data_i(mem2wbk_aligned_data),
    .p_data_i(mem2wbk_p_data),

    .csr_we_i(mem2wbk_csr_we),
    .csr_we_addr_i(mem2wbk_csr_addr),
    .csr_we_data_i(mem2wbk_csr_data),

    // to Register File and Forwarding Unit
    .rd_we_o(wbk_rd_we),
    .rd_addr_o(wbk_rd_addr),
    .rd_data_o(wbk_rd_data),

    // PC of the current instruction.
    .pc_i(mem2wbk_pc),
    .pc_o(wbk2csr_pc),

    // System Jump operations
    .sys_jump_i(mem2wbk_sys_jump),
    .sys_jump_o(wbk2csr_sys_jump),
    .sys_jump_csr_addr_i(mem2wbk_sys_jump_csr_addr),
    .sys_jump_csr_addr_o(wbk2csr_sys_jump_csr_addr),

    // Has instruction fetch being successiful?
    .fetch_valid_i(mem2wbk_fetch_valid),
    .fetch_valid_o(wbk2csr_fetch_valid),

    // to CSR and FWD
    .csr_we_o(wbk_csr_we),
    .csr_we_addr_o(wbk_csr_addr),
    .csr_we_data_o(wbk_csr_data),

    // Exception info passed from Memory to CSR.
    .xcpt_valid_i(mem2wbk_xcpt_valid),
    .xcpt_cause_i(mem2wbk_xcpt_cause),
    .xcpt_tval_i(mem2wbk_xcpt_tval),
    .xcpt_valid_o(wbk2csr_xcpt_valid),
    .xcpt_cause_o(wbk2csr_xcpt_cause),
    .xcpt_tval_o(wbk2csr_xcpt_tval)
);

// =============================================================================
csr_file #( .HART_ID(HART_ID) )
CSR(
    // Top-level system signals
    .clk_i(clk_i),
    .rst_i(rst_i),

    // from Decode
    .csr_raddr_i(dec2csr_csr_addr),

    // to Decode
    .csr_data_o(csr2dec_data),

    // from Writeback
    .csr_we_i(wbk_csr_we),
    .csr_waddr_i(wbk_csr_addr),
    .csr_wdata_i(wbk_csr_data),

    // Interrupts
    .ext_irq_i(ext_irq_i & irq_enable),
    .tmr_irq_i(tmr_irq_i & irq_enable),
    .sft_irq_i(sft_irq_i & irq_enable),
    .irq_taken_o(csr_irq_taken),
    .pc_handler_o(csr_pc_handler),
    .nxt_unwb_PC_i(nxt_unwb_PC),

    // PC of the current instruction.
    .pc_i(wbk2csr_pc),
    .exe_pc_i(exe2mem_pc),
    .stall_i(stall_instr_fetch | stall_data_fetch),

    // System Jump operation
    .sys_jump_i(wbk2csr_sys_jump),
    .sys_jump_csr_addr_i(wbk2csr_sys_jump_csr_addr),
    .sys_jump_csr_data_o(csr_sys_jump_data),
    .sys_jump_o(csr_sys_jump),

    // Current preivilege level
    .privilege_level_o(privilege_level),

    // Exception requests
    .xcpt_valid_i(wbk2csr_xcpt_valid),
    .xcpt_cause_i(wbk2csr_xcpt_cause),
    .xcpt_tval_i(wbk2csr_xcpt_tval)
);

endmodule

    
