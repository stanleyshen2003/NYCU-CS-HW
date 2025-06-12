module ALU( result, zero, overflow, aluSrc1, aluSrc2, invertA, invertB, operation );
   
  output wire[31:0] result;
  output wire zero;
  output wire overflow;

  input wire[31:0] aluSrc1;
  input wire[31:0] aluSrc2;
  input wire invertA;
  input wire invertB;
  input wire[1:0] operation;
  
  /*your code here*/
  wire [31:0] temp_result;
  wire zero_for_set;
  wire [31:0] carry_temp;
  wire set;

  assign zero_for_set = 1'b0;
  wire carry_in_first;
  assign carry_in_first = (invertB==1) ? 1'b1:1'b0;
  
  ALU_1bit ALU_0(.result(temp_result[0]), .carryOut(carry_temp[0]), .a(aluSrc1[0]), .b(aluSrc2[0]), .invertA(invertA), .invertB(invertB), .operation(operation), .carryIn(carry_in_first), .less(set));
	
  genvar index;
  generate
  for (index = 1; index <= 31; index = index + 1)
  begin
    ALU_1bit alu(.result(temp_result[index]), .carryOut(carry_temp[index]), .a(aluSrc1[index]), .b(aluSrc2[index]), .invertA(invertA), .invertB(invertB), .operation(operation), .carryIn(carry_temp[index-1]), .less(zero_for_set));
  end
  endgenerate
  wire add_result,carryOut_temp;
  
  wire sr2_invert;
  assign sr2_invert = ~aluSrc2[31];
  Full_adder adder1(.sum(add_result), .carryOut(carryOut_temp), .carryIn(carry_temp[30]), .input1(aluSrc1[31]), .input2(sr2_invert));
  
  assign set = add_result;
  assign overflow = (operation==2'b01 || operation==2'b00)? 1'b0 : (carry_temp[30] ^ carry_temp[31]);
  
  assign result = temp_result;
  assign zero=(result==32'b0)?1'b1:1'b0;
endmodule