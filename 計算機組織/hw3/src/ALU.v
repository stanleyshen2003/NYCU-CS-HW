module ALU( aluSrc1, aluSrc2, ALU_operation_i, result, zero, overflow );

//I/O ports 
input	[32-1:0] aluSrc1;
input	[32-1:0] aluSrc2;
input	 [4-1:0] ALU_operation_i;

output	[32-1:0] result;
output			 zero;
output			 overflow;

//Internal Signals
wire			 zero;
wire			 overflow;
wire	[32-1:0] result;

//Main function
/*your code here*/


wire invertA;
wire invertB;
wire[1:0] operation;
wire less;

assign invertA = ALU_operation_i[3];
assign invertB = ALU_operation_i[2];
assign operation = ALU_operation_i[1:0];


wire set;
  wire[31:0] carryoutArray;


  genvar i;
  generate
      for(i=1;i<=31;i=i+1) begin : generateBlockAlu31
      ALU_1bit alu(
		  .result(result[i]),
		  .carryOut(carryoutArray[i]),
		  .a(aluSrc1[i]),
		  .b(aluSrc2[i]),
		  .invertA(invertA),
		  .invertB(invertB),
		  .operation(operation),
		  .carryIn(carryoutArray[i-1]),
		  .less(1'b0)
	  );
  end
  endgenerate



  assign set=(aluSrc1==aluSrc2)?0:
  			(aluSrc1[31]==1'b0&&aluSrc2[31]==1'b0&&aluSrc1<aluSrc2)?1:
			(aluSrc1[31]==1'b1&&aluSrc2[31]==1'b0)?1:
			(aluSrc1[31]==1'b1&&aluSrc2[31]==1'b1&&aluSrc2-aluSrc1>0)?1:0;

  assign overflow = carryoutArray[30]^carryoutArray[31];
  assign zero= (result==32'b0) ? 1 : 0 ;


  


   ALU_1bit alu(
    .result(result[0]),
    .carryOut(carryoutArray[0]),
    .a(aluSrc1[0]),
    .b(aluSrc2[0]),
    .invertA(invertA),
    .invertB(invertB),
    .operation(operation),
    .carryIn(invertB),
    .less(set)
    );
  


  ////////
  


endmodule
