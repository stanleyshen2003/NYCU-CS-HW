module Program_Counter( clk_i, rst_n, pc_in_i, pc_out_o );

//I/O ports
input           clk_i;
input	        rst_n;
input  [32-1:0] pc_in_i;
output [32-1:0] pc_out_o;
 
//Internal Signals
reg    [32-1:0] pc_out_o;

//Main function
always @(posedge clk_i) begin
    if(~rst_n)
	    pc_out_o <= 0;
	else
	    pc_out_o <= pc_in_i;
end

endmodule
