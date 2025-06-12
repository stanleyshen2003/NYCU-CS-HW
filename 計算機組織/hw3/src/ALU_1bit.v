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
  wire assign_a,assign_b;
  wire addSum;
  assign assign_a= (invertA==1'b0) ? a : ~a;
  assign assign_b = (invertB==1'b0) ? b : ~b;
 
  Full_adder F(
    .sum(addSum),
    .carryOut(carryOut),
    .carryIn(carryIn),
    .input1(assign_a),
    .input2(assign_b)
    );


  assign addAnd = assign_a & assign_b;
  assign addOR = assign_a | assign_b;

  assign result= (operation==2'b10) ? addSum :
        (operation==2'b00) ? addAnd :
        (operation==2'b01) ? addOR :
        (operation==2'b11) ? less :
        0;

  
endmodule