`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/05/2024 06:40:11 PM
// Design Name: 
// Module Name: Clock_counter
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module profiler (
    input               clk_i,        // Clock input
    input               rst_i,        // Reset input
    input  [31:0]      pc_addr_i,    // Input for program counter address
    input              mem_i,
    input              stall_i
);
    // State to indicate if counting is active
    reg counting;
    reg [31:0]   cycle_count;
    reg [31:0]   core_list_find_count;
    reg [31:0]   core_list_reverse_count;
    reg [31:0]   core_state_transition_count;
    reg [31:0]   matrix_mul_count;
    reg [31:0]   crcu8_count;
    reg [31:0]   mem_core_list_find_count;
    reg [31:0]   mem_core_list_reverse_count;
    reg [31:0]   mem_core_state_transition_count;
    reg [31:0]   mem_matrix_mul_count;
    reg [31:0]   mem_crcu8_count;
    reg [31:0]   stall_core_list_find_count;
    reg [31:0]   stall_core_list_reverse_count;
    reg [31:0]   stall_core_state_transition_count;
    reg [31:0]   stall_matrix_mul_count;
    reg [31:0]   stall_crcu8_count;
    

    always @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            cycle_count <= 32'd0;  // Reset cycle count to 0
            counting <= 1'b0;         // Stop counting on reset
            core_list_find_count <= 32'd0;
            core_list_reverse_count <= 32'd0;
            core_state_transition_count <= 32'd0;
            matrix_mul_count <= 32'd0;
            crcu8_count <= 32'd0;
            mem_core_list_find_count <= 32'd0;
            mem_core_list_reverse_count <= 32'd0;
            mem_core_state_transition_count <= 32'd0;
            mem_matrix_mul_count <= 32'd0;
            mem_crcu8_count <= 32'd0;
            stall_core_list_find_count <= 32'd0;
            stall_core_list_reverse_count <= 32'd0;
            stall_core_state_transition_count <= 32'd0;
            stall_matrix_mul_count <= 32'd0;
            stall_crcu8_count <= 32'd0;
        end 
        else begin
            // Start counting when pc_addr is 0x800
            if (pc_addr_i == 32'h00001088) begin
                counting <= 1'b1;      // Start counting
            end
            // Stop counting when pc_addr is 0x950
            if (pc_addr_i == 32'h00001798) begin
                counting <= 1'b0;      // Stop counting
            end

            // Increment cycle count if counting is active
            if (counting) begin
                cycle_count = cycle_count + 1; // Increment cycle count
            end
        end
    end
    
    always @(posedge clk_i) begin
        if (pc_addr_i >= 32'h00001cfc && pc_addr_i <= 32'h00001d4c) begin
            core_list_find_count <= core_list_find_count + 1;
            if (mem_i) begin
                mem_core_list_find_count <= mem_core_list_find_count +1;
                if (stall_i) begin
                    stall_core_list_find_count <= stall_core_list_find_count +1;
                end
            end
            
        end
        else if (pc_addr_i >= 32'h00001d50 && pc_addr_i <= 32'h00001d70) begin
            core_list_reverse_count <= core_list_reverse_count + 1;
            if (mem_i) begin
                mem_core_list_reverse_count <= mem_core_list_reverse_count + 1;
                if (stall_i) begin
                    stall_core_list_reverse_count <= stall_core_list_reverse_count + 1;
                end 
            end
            
        end
        else if (pc_addr_i >= 32'h000029f4 && pc_addr_i <= 32'h00002cdc) begin
            core_state_transition_count <= core_state_transition_count + 1;
            if (mem_i) begin
                mem_core_state_transition_count <= mem_core_state_transition_count +1;
                if (stall_i) begin
                    stall_core_state_transition_count <= stall_core_state_transition_count +1;
                end
            end 
        end
        else if (pc_addr_i >= 32'h00002650 && pc_addr_i <= 32'h0000270c) begin
            matrix_mul_count <= matrix_mul_count + 1;
            if (mem_i) begin
                mem_matrix_mul_count <= mem_matrix_mul_count + 1;
                if (stall_i) begin
                    stall_matrix_mul_count <= stall_matrix_mul_count + 1;
                end
            end
        end
        else if (pc_addr_i >= 32'h000019b4 && pc_addr_i <= 32'h000019f8) begin
            crcu8_count <= crcu8_count + 1;
            if (mem_i) begin
                mem_crcu8_count <= mem_crcu8_count + 1;
                if (stall_i) begin
                    stall_crcu8_count <= stall_crcu8_count + 1;
                end
            end
            
        end
    end
endmodule

