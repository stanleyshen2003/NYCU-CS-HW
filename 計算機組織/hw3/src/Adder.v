module Adder( src1_i, src2_i, sum_o	);

//I/O ports
input	[32-1:0] src1_i;
input	[32-1:0] src2_i;
output	[32-1:0] sum_o;

//Internal Signals
wire	[32-1:0] sum_o;
    
//Main function
/*your code here*/
wire t2, t3;
wire [3:0] t1;
assign t1 = 4'b0010;
ALU alu1( src1_i, src2_i, t1, sum_o, t2, t3);

endmodule
