module ALU_Ctrl( funct_i, ALUOp_i, ALU_operation_o, FURslt_o );

//I/O ports 
input      [6-1:0] funct_i;
input      [3-1:0] ALUOp_i;

output     [4-1:0] ALU_operation_o;  
output     [2-1:0] FURslt_o;
     
//Internal Signals
wire		[4-1:0] ALU_operation_o;
wire		[2-1:0] FURslt_o;

//Main function
/*your code here*/
reg [4-1:0] ALU_operation_o1;
reg [2-1:0] FURslt_o1;
always@(funct_i, ALUOp_i)
begin
  if(funct_i == 6'b010010 && ALUOp_i == 3'b010)//add
    begin
  	ALU_operation_o1 = 4'b0010;
    FURslt_o1 = 2'b00;
    end
  else if(funct_i == 6'b010000 && ALUOp_i == 3'b010)//sub
    begin
  	ALU_operation_o1 = 4'b0110;
	FURslt_o1 = 2'b00;
    end
  else if(funct_i == 6'b010100 && ALUOp_i == 3'b010)//and
    begin
  	ALU_operation_o1 = 4'b0000;
	FURslt_o1 = 2'b00;
    end
  else if(funct_i == 6'b010110 && ALUOp_i == 3'b010)//or
    begin
  	ALU_operation_o1 = 4'b0001;
	FURslt_o1 = 2'b00;
    end
  else if(funct_i == 6'b010101 && ALUOp_i == 3'b010)//nor
    begin
  	ALU_operation_o1 = 4'b1100;
	FURslt_o1 = 2'b00;
    end
  else if(funct_i == 6'b100000 && ALUOp_i == 3'b010)//slt
    begin
  	ALU_operation_o1 = 4'b0111;
	FURslt_o1 = 2'b00;
	//$display("slt");
    end
  else if(funct_i == 6'b000000 && ALUOp_i == 3'b010)//sll
    begin
  	ALU_operation_o1 = 4'b0001;
	FURslt_o1 = 2'b01;
    end
  else if(funct_i == 6'b000010 && ALUOp_i == 3'b010)//srl
    begin
  	ALU_operation_o1 = 4'b0000;
	FURslt_o1 = 2'b01;
    end
  else if(ALUOp_i == 3'b011)//addi
    begin
	ALU_operation_o1 = 4'b0010;
	FURslt_o1 = 2'b00;
    end
  else if(funct_i == 6'b000110 && ALUOp_i == 3'b010)//sllv
    begin
	ALU_operation_o1 = 4'b0011;
	FURslt_o1 = 2'b01;
    end
  else if(funct_i == 6'b000100 && ALUOp_i == 3'b010)//srlv
    begin
	ALU_operation_o1 = 4'b0010;
	FURslt_o1 = 2'b01;
	end
  else if(ALUOp_i == 3'b000)//lw sw
    begin
	ALU_operation_o1 = 4'b0010;
	FURslt_o1 = 2'b00;
    end
  else if(ALUOp_i == 3'b001)//beq
    begin
	ALU_operation_o1 = 4'b0110;
	FURslt_o1 = 2'b00;
	//$display("beq");
	end
  else if(ALUOp_i == 3'b110)//bne
    begin
	ALU_operation_o1 = 4'b0110;
	FURslt_o1 = 2'b00;
	end
  else if(ALUOp_i == 3'b100)//blt
    begin
	ALU_operation_o1 = 4'b1001;
	FURslt_o1 = 2'b00;
	end
  else if(ALUOp_i == 3'b101)//bnez
    begin
	ALU_operation_o1 = 4'b0110;
	FURslt_o1 = 2'b00;
	end
  else if(ALUOp_i == 3'b111)//bgez
    begin
	ALU_operation_o1 = 4'b1000;
	FURslt_o1 = 2'b00;
	end
  else
	FURslt_o1 = 2'b00;
end 

assign ALU_operation_o = ALU_operation_o1;
assign FURslt_o = FURslt_o1;

endmodule     
