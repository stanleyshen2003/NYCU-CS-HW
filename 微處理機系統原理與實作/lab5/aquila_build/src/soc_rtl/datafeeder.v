`timescale 1ns / 1ps
`include "aquila_config.vh"

module data_feeder #(parameter XLEN = 32) (
    input                           clk_i,
    input                           rst_i,

    // from aquila
    input                           S_DEVICE_strobe_i, 
    input [XLEN-1 : 0]              S_DEVICE_addr_i,
    input                           S_DEVICE_rw_i,
    input [XLEN-1 : 0]              S_DEVICE_data_i,

    // to aquila
    output reg                      S_F_DEVICE_ready_o,
    output reg [XLEN-1 : 0]         S_F_DEVICE_data_o,

    // to floating point IP
    output                          a_valid_o,
    output [XLEN-1 : 0]             a_data_o,
    output                          b_valid_o,
    output [XLEN-1 : 0]             b_data_o,
    output                          c_valid_o,
    output [XLEN-1 : 0]             c_data_o,

    // from floating point IP
    input                           r_valid_i,
    input [XLEN-1 : 0]              r_data_i
);

// =========================================
// Local Parameters
// =========================================
localparam OUPTUT_ADDR     = 32'hC4000000;
localparam DSA_IN1_ADDR    = 32'hC4000004;
localparam DSA_IN2_ADDR    = 32'hC4000008;
localparam DSA_IN3_ADDR    = 32'hC400000C;

// =========================================
// Registers for sequential data loading
// =========================================
reg [XLEN-1 : 0] data_buffer_a;
reg [XLEN-1 : 0] data_buffer_b;
reg [XLEN-1 : 0] data_buffer_c;

// =========================================
// Control signals
// =========================================
assign a_valid_o = 1;
assign b_valid_o = 1;
assign c_valid_o = 1;
assign a_data_o  = data_buffer_a;
assign b_data_o = data_buffer_b;
assign c_data_o  = data_buffer_c;

// =========================================
// Data loading logic
// =========================================
always @(posedge clk_i) begin
    if (rst_i) begin
        data_buffer_a   <= 0;
        data_buffer_b   <= 0;
        data_buffer_c   <= 0;
    end else if (S_DEVICE_strobe_i) begin
        if (S_DEVICE_rw_i) begin
            // Write operations
            case (S_DEVICE_addr_i)
                DSA_IN1_ADDR: begin
                    data_buffer_a <= S_DEVICE_data_i;
                end
                DSA_IN2_ADDR: begin
                    data_buffer_b <= S_DEVICE_data_i;
                end
                DSA_IN3_ADDR: begin
                    data_buffer_c <= S_DEVICE_data_i;
                end
            endcase
        end else begin
            // Read operations
            if (S_DEVICE_addr_i == OUPTUT_ADDR) begin
//                S_F_DEVICE_ready_o <= r_valid_i;
                S_F_DEVICE_data_o  <= r_data_i;
            end
        end
    end 
    
end

// // =========================================
// // S_DEVICE Ready Logic
// // =========================================
 always @(posedge clk_i) begin
     if (rst_i) begin
         S_F_DEVICE_ready_o <= 0;
     end else begin
         S_F_DEVICE_ready_o <= r_valid_i;
     end
 end

endmodule
