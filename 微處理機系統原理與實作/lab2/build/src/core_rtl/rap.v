`timescale 1ns / 1ps
// =============================================================================
//  Program : bpu.v
//  Author  : Jin-you Wu
//  Date    : Jan/19/2019
// -----------------------------------------------------------------------------
//  Description:
//  This is the Branch Prediction Unit (BPU) of the Aquila core (A RISC-V core).
// -----------------------------------------------------------------------------
//  Revision information:
//
//  Aug/15/2020, by Chun-Jen Tsai:
//    Using a single BPU to handle both JAL and Branch instructions. In the
//    original code, an additional Unconditional Branch Prediction Unit (UC-BPU)
//    was used to handle the JAL instructions.
//
// Aug/16/2023, by Chun-Jen Tsai:
//    Replace the fully associative BHT by a TAG-based direct-mapping BHT table.
//    The 2-bit Bimodal FSM is still used. The performance drops a little (0.03
//    DMIPS), but the resource usage drops significantly. Note that we do not
//    have to use TAGged memory for direct-mapping BHT. We simply use it to
//    demonstrate how TAG memory can be implemented.
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

module rap #( parameter ENTRY_NUM = 64, parameter XLEN = 32, parameter BUFFER_SIZE = 32 ) //64
(
    // System signals
    input               clk_i,
    input               rst_i,
    input               stall_i,

    // from Program_Counter
    input  [XLEN-1 : 0] pc_i, // Addr of the next instruction to be fetched.

    // from Decode
    input               dec_is_jalr_i, // For cache miss, update buffer
    input               dec_is_ret_i,
    input               dec_is_jal_i, // Update circular buffer
    input  [XLEN-1 : 0] dec_pc_i, // Update circular buffer Addr of the instr. just processed by decoder.
    input               dec_addr_change_i,
    input               dec_is_branch_i,
    // from Execute
    input               exe_is_jalr_i,
    input               exe_is_jal_i,
    input               exe_is_ret_i,
    input               exe_is_branch_i,
    input  [XLEN-1 : 0] exe_pc_i, // branch
    input               exe_is_jr_i,

    // to Program_Counter
    output              ret_hit_o,
    output [XLEN-1 : 0] ret_target_addr_o
);

localparam NBITS = $clog2(ENTRY_NUM);

wire [NBITS-1 : 0]      read_addr;
wire [NBITS-1 : 0]      write_addr;
wire [XLEN-1 : 0]       ret_inst_tag;
wire                    we;
reg                     BHT_hit_ff, BHT_hit;
reg  [31:0]             ret_hit, ret_miss;
wire                    push, pop, buffer_empty, buffer_full;
wire [XLEN -1:0]        buffer_out, buffer_in;
reg                     enable;
// two-bit saturating counter
//reg  [1 : 0]            branch_likelihood[ENTRY_NUM-1 : 0];

// "we" is enabled to add a new entry to the BHT table when
// the decoded branch instruction is not in the BHT.
// CY Hsiang 0220_2020: added "~stall_i" to "we ="
assign we = enable & ~stall_i & dec_is_ret_i & ~BHT_hit;// & ~(exe_is_jalr_i | exe_is_jal_i | exe_is_branch_i);
// pop if hit or (not hit and is ret)
assign pop = enable & ~stall_i & exe_is_ret_i; 
assign buffer_in = exe_pc_i + 4;
assign read_addr = pc_i[NBITS+2 : 2];
assign write_addr = dec_pc_i[NBITS+2 : 2];
assign push = enable & (exe_is_jal_i | (exe_is_jalr_i & ~exe_is_ret_i)) & ~exe_is_jr_i & ~stall_i;
reg counting;


// ret analysis
always @(posedge clk_i)
begin
    
    if (rst_i)
    begin
        ret_hit <= 0;
        ret_miss <= 0;
        enable <= 0;
        counting <= 0;
    end
    
    if (pc_i == 32'h00001088) begin
        counting <= 1'b1;      // Start counting
        enable <= 1'b1;
    end
            // Stop counting when pc_addr is 0x950
    if (pc_i == 32'h000017a4) begin
        counting <= 1'b0;      // Stop counting
        enable <= 1'b0;
    end
    if (dec_is_ret_i && counting)
    begin
        if(BHT_hit)
        begin
            ret_hit <= ret_hit + 1;
        end
        else
        begin
            ret_miss <= ret_miss + 1;
        end
    end        
end


// ===========================================================================
//  Branch History Table (BHT). Here, we use a direct-mapping cache table to
//  store branch history. Each entry of the table contains two fields:
//  the branch_target_addr and the PC of the branch instruction (as the tag).
//
distri_ram #(.ENTRY_NUM(ENTRY_NUM), .XLEN(XLEN))
RAP_HT(
    .clk_i(clk_i),
    .we_i(we),                  // Write-enabled when the instruction at the Decode
                                //   is a branch and has never been executed before.
    .write_addr_i(write_addr),  // Direct-mapping index for the branch at Decode.
    .read_addr_i(read_addr),    // Direct-mapping Index for the next PC to be fetched.

    .data_i(dec_pc_i), // Input is not used when 'we' is 0.
    .data_o(ret_inst_tag)
);

circular_filo_buffer #(.XLEN(XLEN), .BUFFER_SIZE(BUFFER_SIZE))
RAB_BUFFER(
    .clk_i(clk_i),
    .rst_i(rst_i),
    .push_i(push),               // Push data onto the stack
    .pop_i(pop),                 // Pop data from the stack
    .data_in_i(buffer_in),
    .data_out_o(buffer_out),
    .empty_o(buffer_empty)              // Buffer is empty
);

// Delay the BHT hit flag at the Fetch stage for two clock cycles (plus stalls)
// such that it can be reused at the Execute stage for BHT update operation.
always @ (posedge clk_i)
begin
    if (rst_i) begin
        BHT_hit_ff <= 1'b0;
        BHT_hit <= 1'b0;
    end
    else if (!stall_i) begin
        BHT_hit_ff <= ret_hit_o;
        BHT_hit <= BHT_hit_ff;
    end
end

// ===========================================================================
//  Outputs signals
//
assign ret_hit_o = (ret_inst_tag == pc_i) & ~buffer_empty; // & ~(dec_is_jal_i) & ~(dec_is_jalr_i) & ~(dec_is_branch_i) & ~(exe_is_jal_i) & ~(exe_is_jalr_i) & ~(exe_is_branch_i);
assign ret_target_addr_o = buffer_out;
//assign branch_decision_o = branch_likelihood[read_addr][1];

endmodule
