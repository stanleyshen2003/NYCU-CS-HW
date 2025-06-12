module circular_filo_buffer
#(parameter XLEN = 32, 
  parameter BUFFER_SIZE = 32)
(
    input wire clk_i,
    input wire rst_i,
    input wire push_i,                // Push data onto the stack
    input wire pop_i,                 // Pop data from the stack
    input wire [XLEN -1:0] data_in_i,
    output wire [XLEN -1:0] data_out_o,
    output wire empty_o              // Buffer is empty
);

localparam BUFFER_IDX = $clog2(BUFFER_SIZE);
    reg [XLEN-1:0] buffer [BUFFER_SIZE-1:0]; // Buffer memory
    reg [BUFFER_IDX + 1:0] top_ptr;                             // Pointer for FILO stack
    reg [BUFFER_IDX + 1:0] item_count;                          // Count of items in the buffer
    wire full;
    // Full and Empty conditions
    assign empty_o = (item_count == 0);
    assign full = (item_count == BUFFER_SIZE);
    assign data_out_o = (top_ptr == 0)? 32'b0:buffer[(top_ptr+BUFFER_SIZE-1)%BUFFER_SIZE];
    always @(posedge clk_i or posedge rst_i) begin
        if (rst_i) begin
            top_ptr <= 0;
            item_count <= 0;
        end else begin
            if (push_i) begin
                buffer[top_ptr] <= data_in_i;       // Write data to the current position
                
                // Increment the top_ptr in a circular manner
                top_ptr <= (top_ptr + 1) % BUFFER_SIZE;

                // If the buffer is full, keep item_count at BUFFER_SIZE, otherwise increment it
                if (!full) begin
                    item_count <= item_count + 1;
                end
            end
            if (pop_i && !empty_o) begin
                // Decrement top_ptr in a circular manner and read data
                top_ptr <= (top_ptr + BUFFER_SIZE -1) % BUFFER_SIZE;
                item_count <= item_count - 1;
            end
        end
    end
endmodule
