`timescale 1ns / 1ps
// =============================================================================
//  Program : core2axi_if.v
//  Author  : Chun-Jen Tsai
//  Date    : Dec/22/2022
// -----------------------------------------------------------------------------
//  Description:
//  This is the AXI bus interface for the Aquila core. This module converts
//  the Aquila device I/O interface to the AXI-lite bus master interface such
//  that you can connect it to an AXI-lite salve device.
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
//  Copyright 2022,
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

module core2axi_if #(parameter XLEN = 32, AXI_ADDR_LEN = 8, AXI_DATA_LEN = 32)
(
    input                 clk_i,
    input                 rst_i,   // level-sensitive reset signal.

    // Aquila M_DEVICE master interface signals.
    input                 S_DEVICE_strobe_i,
    input  [XLEN-1 : 0]   S_DEVICE_addr_i,
    input                 S_DEVICE_rw_i,
    input  [XLEN/8-1 : 0] S_DEVICE_byte_enable_i,
    input  [XLEN-1 : 0]   S_DEVICE_data_i,
    output                S_DEVICE_data_ready_o,
    output [XLEN-1 : 0]   S_DEVICE_data_o,

    // Converted AXI master interface signals.
    output reg [AXI_ADDR_LEN - 1 : 0] m_axi_awaddr,  // Master write address signals.
    output reg                        m_axi_awvalid, // Master write addr/ctrl is valid.
    input                             m_axi_awready, // Slave ready to receive write command.
    output [AXI_DATA_LEN-1 : 0]       m_axi_wdata,   // Master write data signals.
    output [AXI_DATA_LEN/8 - 1 : 0]   m_axi_wstrb,   // Master byte select signals.
    output reg                        m_axi_wvalid,  // Master write data is valid.
    input                             m_axi_wready,  // Slave ready to receive write data.
    input  [1 : 0]                    m_axi_bresp,   // Slave write-op response signal.
    input                             m_axi_bvalid,  // Slave write-op response is valid.
    output reg                        m_axi_bready,  // Master ready to receive write response.
    output reg [AXI_ADDR_LEN - 1 : 0] m_axi_araddr,  // Master read address signals.
    output reg                        m_axi_arvalid, // Master read addr/ctrl is valid.
    input                             m_axi_arready, // Slave is ready to receive read command.
    input  [AXI_DATA_LEN - 1 : 0]     m_axi_rdata,   // Slave read data signals.
    input  [1 : 0]                    m_axi_rresp,   // Slave read-op response signal
    input                             m_axi_rvalid,  // Slave read response is valid.
    output reg                        m_axi_rready   // Master ready to receive read response.
);

wire               write_resp_error, read_resp_error;
reg                read_done, write_done;
reg  [XLEN-1 : 0]  dev_rdata;

assign m_axi_wdata = S_DEVICE_data_i;
assign m_axi_wstrb = S_DEVICE_byte_enable_i;
assign S_DEVICE_data_ready_o = read_done || write_done;
assign S_DEVICE_data_o = dev_rdata;

// Check for write_done completion.
always @(posedge clk_i)
begin
    if (m_axi_bvalid && m_axi_bready) // The write_done should be associated
        write_done <= 1;              // with a bready response.
    else
        write_done <= 0;
end

// Check for read_done completion.
always @(posedge clk_i)
begin
    if (m_axi_rvalid && m_axi_rready) // The read_done should be associated
    read_done <= 1;                   // with a rready response.
else
    read_done <= 0;
end

// Flag any write errors.
assign write_resp_error = (m_axi_bready & m_axi_bvalid & m_axi_bresp[1]);

// Flag any read errors.
assign read_resp_error = (m_axi_rready & m_axi_rvalid & m_axi_rresp[1]);

// -----------------------
//  Write Address Channel
// -----------------------
always @(posedge clk_i)
begin
    if (rst_i)
        m_axi_awvalid <= 0;
    else if (S_DEVICE_strobe_i & S_DEVICE_rw_i)
        m_axi_awvalid <= 1;
    else if (m_axi_awvalid & m_axi_awready)
        m_axi_awvalid <= 0;
    else
        m_axi_awvalid <= m_axi_awvalid;
end

always @(posedge clk_i) // Write Addresses
begin
    if (rst_i)
        m_axi_awaddr <= 0;
    else if (S_DEVICE_strobe_i & S_DEVICE_rw_i)
        m_axi_awaddr <= S_DEVICE_addr_i[AXI_ADDR_LEN - 1 : 0];
end

// --------------------
//  Write Data Channel
// --------------------
always @(posedge clk_i)
begin
    if (rst_i)
        m_axi_wvalid <= 0;
    else if (S_DEVICE_strobe_i & S_DEVICE_rw_i)
        m_axi_wvalid <= 1;
    else if (m_axi_wready & m_axi_wvalid)  // Data accepted by the slave
        m_axi_wvalid <= 0;                 // (slave issued M_AXI_WREADY).
    else
        m_axi_wvalid <= m_axi_wvalid;
end

// ----------------------------
//  Write Response (B) Channel
// ----------------------------
always @(posedge clk_i)
begin
    if (rst_i)
        m_axi_bready <= 0;
    else if (m_axi_bvalid & ~m_axi_bready) // Accept/ack. bresp with m_axi_bready
        m_axi_bready <= 1;                 // by the master when m_axi_bvalid
    else if (m_axi_bready)                 // is asserted by slave.
        m_axi_bready <= 0;                 // Deassert after one clock cycle.
    else
        m_axi_bready <= m_axi_bready;      // Retain the previous value.
end

// ----------------------
//  Read Address Channel
// ----------------------
always @(posedge clk_i)
begin
    if (rst_i)
        m_axi_arvalid <= 0;
    else if (S_DEVICE_strobe_i & ~S_DEVICE_rw_i)
        m_axi_arvalid <= 1;
    else if (m_axi_arready & m_axi_arvalid)
        m_axi_arvalid <= 0;
    else
        m_axi_arvalid <= m_axi_arvalid;
end

always @(posedge clk_i) // Read Addresses
begin
    if (rst_i)
        m_axi_araddr <= 0;
    else if (S_DEVICE_strobe_i & ~S_DEVICE_rw_i)
        m_axi_araddr <= S_DEVICE_addr_i[AXI_ADDR_LEN - 1 : 0];
end

// ----------------------------------
//  Read Data (and Response) Channel
// ----------------------------------
always @(posedge clk_i)
begin
    if (rst_i)
        m_axi_rready <= 0;
    else if (m_axi_rvalid & ~m_axi_rready) // Accept/ack. rdata/rresp with
        m_axi_rready <= 1;                 // m_axi_rready by the master when
    else if (m_axi_rready)                 // M_AXI_RVALID is asserted by slave.
        m_axi_rready <= 0;                 // Deassert after one clock cycle.
    else
        m_axi_rready <= m_axi_rready;
end

always @(posedge clk_i)
begin
    if (rst_i)
        dev_rdata <= {XLEN{1'b0}};
    else if (m_axi_rvalid)
        dev_rdata <= {{(XLEN-AXI_DATA_LEN){1'b0}}, m_axi_rdata };
    else
        dev_rdata <= dev_rdata;
end

endmodule
