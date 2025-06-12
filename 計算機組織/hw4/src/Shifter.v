module Shifter( result, leftRight, shamt, sftSrc );

//I/O ports 
output	[32-1:0] result;

input			leftRight;
input	[5-1:0] shamt;/*sllv srlv input rs
						sll srl input */
input	[32-1:0] sftSrc ;

//Internal Signals
wire	[32-1:0] result;
  
//Main function

reg [32-1:0] result1;
always@(leftRight or sftSrc or shamt)
begin
  case(leftRight)
    0: result1 = (sftSrc >> shamt);
    1: result1 = (sftSrc << shamt);
  endcase
end
assign result = result1;
endmodule
