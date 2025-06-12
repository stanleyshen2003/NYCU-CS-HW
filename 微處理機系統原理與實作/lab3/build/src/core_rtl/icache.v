`timescale 1ns / 1ps
// =============================================================================
//  Program : icache.v
//  Author  : Jin-you Wu
//  Date    : Oct/31/2018
// -----------------------------------------------------------------------------
//  Description:
//  This module implements the L1 Instruction Cache with the following
//  properties:
//      4-way set associative
//      FIFO replacement policy
//      Read-only
// -----------------------------------------------------------------------------
//  Revision information:
//
//  Dec/06/2022, by Che-Yu Wu:
//    Forward the instruction data when fetching across line index occurs.
//    This design avoids one stall cycle per each cache line crossing fetch.
//
//  Sep/22/2023, by Chun-Jen Tsai:
//    Modify the code to use distributed RAM to store VALID & TAG bits.
//    This modification significantly reduces the resource usage.
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

module icache
#(parameter XLEN = 32,
  parameter CACHE_SIZE = 64,
  parameter CLSIZE = `CLP    // Cache line size.
)
(
    /////////// System signals   ///////////////////////////////////////////////
    input                     clk_i, rst_i,

    /////////// Processor signals //////////////////////////////////////////////
    input                     p_strobe_i,      // Processor request signal.
    input  [XLEN-1 : 0]       p_addr_i,        // Memory addr of the request.
    output reg [XLEN-1 : 0]   p_data_o,        // Data from main memory.
    output                    p_ready_o,       // The cache data is ready.
    input                     p_flush_i,       // Cache flush request.

    /////////// External memory signals   //////////////////////////////////////
    output reg                m_strobe_o,      // Cache request to memory.
    output reg [XLEN-1 : 0]   m_addr_o,        // Address of the request.
    input  [CLSIZE-1 : 0]     m_data_i,        // Data from memory controller.
    input                     m_ready_i,       // Data from memory is ready.

    /////////// Control signals from other caches   ////////////////////////////
    input                     d_flushing_i     // D-Cache is busy flushing
);

//=======================================================
// Cache parameters
//=======================================================
localparam N_WAYS      = 4;
localparam N_LINES     = (CACHE_SIZE*1024*8) / (N_WAYS*CLSIZE);

localparam WAY_BITS    = $clog2(N_WAYS);
localparam BYTE_BITS   = 2;
localparam WORD_BITS   = $clog2(CLSIZE/XLEN);
localparam LINE_BITS   = $clog2(N_LINES);
localparam NONTAG_BITS = LINE_BITS + WORD_BITS + BYTE_BITS;
localparam TAG_BITS    = XLEN - NONTAG_BITS;

//=======================================================
// N-way associative cache signals
//=======================================================
wire                   way_hit[0 : N_WAYS-1];     // Cache-way hit flag.
reg  [WAY_BITS-1 : 0]  hit_index;                 // Decoded way_hit[] signal.
wire                   cache_hit;                 // Got a cache hit?
wire [CLSIZE-1 : 0]    c_block[0 : N_WAYS-1];     // Cache blocks from N cache way.
wire [CLSIZE-1 : 0]    c_data_hit;                // Data from the hit cache block.
reg                    cache_write[0 : N_WAYS-1]; // WE signal for a $ block.
reg                    valid_write[0 : N_WAYS-1]; // WE signal for a $ valid bit.
wire [TAG_BITS-1 : 0]  c_tag_o[0 : N_WAYS-1];     // Tag bits of current $ blocks.
wire                   c_valid_o[0 : N_WAYS-1];   // Validity of current $ blocks.
reg  [LINE_BITS-1 : 0] init_count;                // Counter to initialize valid bits.

assign c_data_hit = c_block[hit_index];

//=======================================================
// FIFO replace policy signals
//=======================================================
reg  [WAY_BITS-1 : 0] FIFO_cnt[0 : N_LINES-1];   // Replace policy counter.
reg  [WAY_BITS-1 : 0] victim_sel;                // The victim cache select.

//=======================================================
// Cache line and tag calculations
//=======================================================
wire [WORD_BITS-1 : 0] line_offset;
wire [LINE_BITS-1 : 0] line_index;
wire [TAG_BITS-1  : 0] tag;

assign line_offset = p_addr_i[WORD_BITS + BYTE_BITS - 1 : BYTE_BITS];
assign line_index  = p_addr_i[NONTAG_BITS - 1 : WORD_BITS + BYTE_BITS];
assign tag         = p_addr_i[XLEN - 1 : NONTAG_BITS];

reg  [LINE_BITS-1 : 0] index_prev;
wire                   is_diff_index;

// Dec/06/2022, by Che-Yu Wu
reg  [XLEN-1 : 0]      p_data_reg;
reg                    is_diff_index_reg;
reg  [WORD_BITS-1 : 0] line_offset_reg;
reg  [XLEN-1 : 0]      fromCache_fwd;
reg  [N_WAYS-1 : 0]    way_offset_reg;
reg  [CLSIZE-1 : 0]    c_data_fwd;

//=======================================================
// Cache Finite State Machine
//=======================================================
localparam Init             = 0,
           Idle             = 1,
           Next             = 2,
           RdfromMem        = 3,
           RdfromMemFinish  = 4;

// Cache controller state register
reg [ 2 : 0] S, S_nxt;

//====================================================
// Cache Controller FSM
//====================================================
always @(posedge clk_i)
begin
    if (rst_i)
        S <= Init;
    else
        S <= S_nxt;
end

always @(*)
begin
    case (S)
        Init: // Multi-cycle initialization of the VALID bits memory.
            if (init_count < N_LINES - 1)
                S_nxt = Init;
            else
                S_nxt = Idle;
        Idle:
            if (p_strobe_i)
                S_nxt = Next;
            else
                S_nxt = Idle;
        Next:
            if (!cache_hit && !d_flushing_i)
                S_nxt =  RdfromMem;
            else
                S_nxt = Next;
        RdfromMem:
            if (m_ready_i)
                S_nxt = RdfromMemFinish;
            else
                S_nxt = RdfromMem;
        RdfromMemFinish:
            S_nxt = Next;
        default:
            S_nxt = Idle;
    endcase
end

// Initialization of the valid bits to zeros upon reset.
always @ (posedge clk_i)
begin
    if (rst_i)
        init_count <= 0;
    else
        init_count <= init_count + (init_count < N_LINES - 1);
end

// Check and see if any cache way has the matched memory block.
assign way_hit[0] = (c_valid_o[0] && (c_tag_o[0] == tag))? 1 : 0;
assign way_hit[1] = (c_valid_o[1] && (c_tag_o[1] == tag))? 1 : 0;
assign way_hit[2] = (c_valid_o[2] && (c_tag_o[2] == tag))? 1 : 0;
assign way_hit[3] = (c_valid_o[3] && (c_tag_o[3] == tag))? 1 : 0;
assign cache_hit  = (way_hit[0] || way_hit[1] || way_hit[2] || way_hit[3]);

always @(*)
begin
    case ( { way_hit[0], way_hit[1], way_hit[2], way_hit[3] } )
        4'b1000: hit_index = 0;
        4'b0100: hit_index = 1;
        4'b0010: hit_index = 2;
        4'b0001: hit_index = 3;
        default: hit_index = 0; // error: multiple-way hit!
    endcase
end

always @(posedge clk_i)
begin
    victim_sel <= FIFO_cnt[line_index];
end

integer idx;

always @(*)
begin
    if (S == RdfromMem && m_ready_i)
        for (idx = 0; idx < N_WAYS; idx = idx + 1)
            cache_write[idx] = (idx == victim_sel);
    else
        for (idx = 0; idx < N_WAYS; idx = idx + 1)
            cache_write[idx] = 1'b0;
end

always @(*)
begin
    if (S == Init)
        for (idx = 0; idx < N_WAYS; idx = idx + 1)
            valid_write[idx] = 1'b1;
    else
        for (idx = 0; idx < N_WAYS; idx = idx + 1)
            valid_write[idx] = cache_write[idx];
end

// Dec/06/2022, by Che-Yu Wu
always @(*)
begin
    case ( way_offset_reg )
        4'b1000: c_data_fwd = c_block[0];
        4'b0100: c_data_fwd = c_block[1];
        4'b0010: c_data_fwd = c_block[2];
        4'b0001: c_data_fwd = c_block[3];
        default: c_data_fwd = 0; // error: multiple-way hit!
    endcase
end

always @(posedge clk_i)
begin
    if (rst_i)
        for (idx = 0; idx < N_LINES; idx = idx + 1)
            FIFO_cnt[idx] <= 0;
    else if (S == RdfromMemFinish)
        FIFO_cnt[line_index] <= FIFO_cnt[line_index] + 1;
end

/* Solution for block ram delay issue */
assign is_diff_index = (index_prev != line_index);

// Dec/06/2022, by Che-Yu Wu
always @(posedge clk_i)
begin
    index_prev <= line_index;
    is_diff_index_reg <= is_diff_index;
    line_offset_reg <= line_offset;
    way_offset_reg <= {way_hit[0], way_hit[1], way_hit[2], way_hit[3]};
end

//-----------------------------------------------
// Read a 32-bit word from the target cache line
//-----------------------------------------------
reg [XLEN-1 : 0] fromCache; // Get the specific word in cache line
reg [XLEN-1 : 0] fromMem;   // Get the specific word in memory line

always @(*)
begin // for hit
    case (line_offset)
`ifdef ARTY
        2'b11: fromCache = c_data_hit[ 31: 0];     // [127: 96]
        2'b10: fromCache = c_data_hit[ 63: 32];    // [ 95: 64]
        2'b01: fromCache = c_data_hit[ 95: 64];    // [ 63: 32]
        2'b00: fromCache = c_data_hit[127: 96];    // [ 31:  0]
`else // KC705
        3'b111: fromCache = c_data_hit[ 31: 0];    // [255:224]
        3'b110: fromCache = c_data_hit[ 63: 32];   // [223:192]
        3'b101: fromCache = c_data_hit[ 95: 64];   // [191:160]
        3'b100: fromCache = c_data_hit[127: 96];   // [159:128]
        3'b011: fromCache = c_data_hit[159: 128];  // [127: 96]
        3'b010: fromCache = c_data_hit[191: 160];  // [ 95: 64]
        3'b001: fromCache = c_data_hit[223: 192];  // [ 63: 32]
        3'b000: fromCache = c_data_hit[255: 224];  // [ 31:  0]
`endif
    endcase
end

always @(*)
begin // for miss
    case (line_offset)
`ifdef ARTY
        2'b11: fromMem = m_data_i[ 31: 0];        // [127: 96]
        2'b10: fromMem = m_data_i[ 63: 32];       // [ 95: 64]
        2'b01: fromMem = m_data_i[ 95: 64];       // [ 63: 32]
        2'b00: fromMem = m_data_i[127: 96];       // [ 31:  0]
`else // KC705
        3'b111: fromMem = m_data_i[ 31: 0];       // [255:224]
        3'b110: fromMem = m_data_i[ 63: 32];      // [223:192]
        3'b101: fromMem = m_data_i[ 95: 64];      // [191:160]
        3'b100: fromMem = m_data_i[127: 96];      // [159:128]
        3'b011: fromMem = m_data_i[159: 128];     // [127: 96]
        3'b010: fromMem = m_data_i[191: 160];     // [ 95: 64]
        3'b001: fromMem = m_data_i[223: 192];     // [ 63: 32]
        3'b000: fromMem = m_data_i[255: 224];     // [ 31:  0]
`endif
    endcase
end

// Dec/06/2022, by Che-Yu Wu
always @(*)
begin // for hit
    case (line_offset_reg)
`ifdef ARTY
        2'b11: fromCache_fwd = c_data_fwd[ 31: 0];     // [127: 96]
        2'b10: fromCache_fwd = c_data_fwd[ 63: 32];    // [ 95: 64]
        2'b01: fromCache_fwd = c_data_fwd[ 95: 64];    // [ 63: 32]
        2'b00: fromCache_fwd = c_data_fwd[127: 96];    // [ 31:  0]
`else // KC705
        3'b111: fromCache_fwd = c_data_fwd[ 31: 0];    // [255:224]
        3'b110: fromCache_fwd = c_data_fwd[ 63: 32];   // [223:192]
        3'b101: fromCache_fwd = c_data_fwd[ 95: 64];   // [191:160]
        3'b100: fromCache_fwd = c_data_fwd[127: 96];   // [159:128]
        3'b011: fromCache_fwd = c_data_fwd[159: 128];  // [127: 96]
        3'b010: fromCache_fwd = c_data_fwd[191: 160];  // [ 95: 64]
        3'b001: fromCache_fwd = c_data_fwd[223: 192];  // [ 63: 32]
        3'b000: fromCache_fwd = c_data_fwd[255: 224];  // [ 31:  0]
`endif
    endcase
end

// Output signals   ////////////////////////////////////////////////////////////
// Delay the output instruction from i-cache until next clock edge.
// This is used to match the behavior of the TCM memory.
// CY Hsiang July 20 2020
// Dec/06/2022, by Che-Yu Wu
always @(posedge clk_i) begin
    if (rst_i)
        p_data_reg <= {(XLEN-1){1'b0}};
    else
        p_data_reg <= ((S == Next) && cache_hit) ? fromCache : (m_ready_i) ? fromMem : 0;
end

always @(*) begin
    p_data_o = (is_diff_index_reg)? fromCache_fwd: p_data_reg;
end

assign p_ready_o = (((S == Next) && cache_hit) || m_ready_i)? 1 : 0;

//======================================================================
// Create a single-cycle memory request pluse for the memory controller
//======================================================================
// The old code uses the reqest/act protocol, which is corrected by the
// CDC synchronizer to match the strobe protocol of MIG. Modified to
// strobe protocol by Chun-Jen Tsai, 09/29/2023.
wire m_strobe;
reg  m_strobe_r;

assign m_strobe = (S == RdfromMem) && !m_ready_i;

always @(posedge clk_i)
    m_strobe_r <= m_strobe;

always @(posedge clk_i)
begin
    if (rst_i)
        m_strobe_o <= 0;
    else if (m_strobe && !m_strobe_r)
        m_strobe_o <= 1;
    else
        m_strobe_o <= 0;
end
//======================================================================

always @(posedge clk_i)
begin
    if (rst_i)
        m_addr_o <= 0;
    else if (S == RdfromMem) // read a cache block
        m_addr_o <= {p_addr_i[XLEN-1 : WORD_BITS+2], {WORD_BITS{1'b0}}, 2'b0};
    else
        m_addr_o <= {XLEN{1'b0}};
end

//=======================================================
//  Cache data storage in Block RAM
//=======================================================
genvar i;
generate
    for (i = 0; i < N_WAYS; i = i + 1)
    begin
        sram #(.DATA_WIDTH(CLSIZE), .N_ENTRIES(N_LINES))
             DATA_BRAM(
                 .clk_i(clk_i),
                 .en_i(1'b1),
                 .we_i(cache_write[i]),
                 .addr_i(line_index),
                 .data_i(m_data_i),  // Data from memory.
                 .data_o(c_block[i])
             );
    end
endgenerate

//=======================================================
//  Tags storage in Distributed RAM
//=======================================================
genvar j;
generate
    for (j = 0; j < N_WAYS; j = j + 1)
    begin
        distri_ram #(.ENTRY_NUM(N_LINES), .XLEN(TAG_BITS))
             TAG_RAM(
                 .clk_i(clk_i),
                 .we_i(cache_write[j]),
                 .read_addr_i(line_index),
                 .write_addr_i(line_index),
                 .data_i(tag),
                 .data_o(c_tag_o[j])
             );
    end
endgenerate

//=======================================================
//  Valid bits storage in Distributed RAM
//=======================================================
genvar k;
generate
    for (k = 0; k < N_WAYS; k = k + 1)
    begin
        distri_ram #(.ENTRY_NUM(N_LINES), .XLEN(1))
             VALID_RAM(
                 .clk_i(clk_i),
                 .we_i(valid_write[k]),
                 .read_addr_i(line_index),
                 .write_addr_i((S == Init)? init_count : index_prev),
                 .data_i(S == RdfromMem && m_ready_i),
                 .data_o(c_valid_o[k])
             );
    end
endgenerate

endmodule
