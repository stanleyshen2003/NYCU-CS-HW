module Sign_Extend( data_i, data_o );

//I/O ports
input	[16-1:0] data_i;
output	[32-1:0] data_o;

//Internal Signals
wire	[32-1:0] data_o;

//Sign extended
/*your code here*/
assign data_o[15:0] = data_i[15:0];
assign data_o[31:16] = (data_i[15] == 1'b1)? 16'hFFFF : 16'h0000;

endmodule      
