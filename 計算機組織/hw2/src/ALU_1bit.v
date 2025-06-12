
module ALU_1bit( result, carryOut, a, b, invertA, invertB, operation, carryIn, less ); 
  
  output wire result;
  output wire carryOut;
  
  input wire a;
  input wire b;
  input wire invertA;
  input wire invertB;
  input wire[1:0] operation;
  input wire carryIn;
  input wire less;
  
  /*your code here*/ 
  wire a_after, b_after, and_result, or_result, add_result;
  xor xor1(a_after,a,invertA);
  xor xor2(b_after,b,invertB);
  or or1(or_result,a_after,b_after);
  and and1(and_result,a_after,b_after);
  
  Full_adder adder1(.sum(add_result), .carryOut(carryOut), .carryIn(carryIn), .input1(a_after), .input2(b_after));
  
  assign result = (operation == 2'b01) ? and_result :
                  (operation == 2'b11) ? less :
                  (operation == 2'b10) ? add_result :
                  or_result;
  
endmodule