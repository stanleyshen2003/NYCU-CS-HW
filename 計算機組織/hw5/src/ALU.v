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
assign overflow = 0;
reg [32-1:0] result1;
reg overflow1 = 0;

always@(aluSrc1, aluSrc2, ALU_operation_i) 
begin
	//$display("alusrc1: %d alusrc2: %d ALU_operation_i:%b",aluSrc1, aluSrc2, ALU_operation_i);
  case(ALU_operation_i)
    4'b0000: result1 = aluSrc1 & aluSrc2;//and
    4'b0001: result1 = aluSrc1 | aluSrc2;//or
    4'b0010: 
      begin
	result1 = aluSrc1 + aluSrc2;//add
	if(aluSrc1 >= 0 && aluSrc2 >= 0 && result1[31] == 1)//p + p = n
	  overflow1 = 1;
	else if(aluSrc1 < 0 && aluSrc2 < 0 && result1[31] == 0)//n + n = p
	  overflow1 = 1;
      end
    4'b0110:
      begin 
	result1 = aluSrc1 - aluSrc2;//sub
	if(aluSrc1 >= 0 && aluSrc2 < 0 && result1[31] == 1)//p - n = n
	  overflow1 = 1;
	else if(aluSrc1 < 0 && aluSrc2 >= 0 && result1[31] == 0)//n - p = p
	  overflow1 = 1;
	 //$display("aluSrc1: %d, aluSrc2: %d, result: %d", aluSrc1, aluSrc2,result1);
      end
    4'b0111: 
      begin
	result1 = aluSrc1 - aluSrc2; //slt
	if(aluSrc1 >= 0 && aluSrc2 < 0 && result1[31] == 1)//p - n = n
	  result1 = 0;
	else if(aluSrc1 < 0 && aluSrc2 >= 0 && result1[31] == 0)//n - p = p
	  result1 = 1;
	else if(result1[31] == 1) 
	  result1 = 1;
	else
	  result1 = 0;
	//$display("aluSrc1: %d, aluSrc2: %d, result: %d", aluSrc1, aluSrc2,result1);
      end
    4'b1100: result1 = ~(aluSrc1 | aluSrc2);//nor
	4'b1000: 
	 begin
	result1 = ($signed(aluSrc1) >= 0) ? 0 : 1; //sget: if greater or equal 0, result = 0
      end
	4'b1001: 
	 begin
	result1 = aluSrc1 - aluSrc2; //slt': result = 1 if rs < rt
	if(aluSrc1 >= 0 && aluSrc2 < 0 && result1[31] == 1)//p - n = n
	  result1 = 0;
	else if(aluSrc1 < 0 && aluSrc2 >= 0 && result1[31] == 0)//n - p = p
	  result1 = 1;
	else if(result1[31] == 1) 
	  result1 = 1;
	else if(result1 == 0)
	  result1 = 1;
	else
	  result1 = 0;
	result1 = (result1 == 0) ? 1 : 0;
      end
    default: result1 = 0;
  endcase
end

assign result = result1;
assign overflow = overflow1;
assign zero = (result == 0) ? 1 : 0;//zero is 1 if result is 0
/*always@(*)
begin
    $display("result: %d", result);
	$display("zero: %d", zero);
end*/
endmodule
