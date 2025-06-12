module Decoder( instr_op_i, RegWrite_o,	ALUOp_o, ALUSrc_o, RegDst_o, Jump_o, Branch_o, BranchType_o, MemWrite_o, MemRead_o, MemtoReg_o);
     
//I/O ports
input	[6-1:0] instr_op_i;

output			RegWrite_o;
output	[3-1:0] ALUOp_o;
output			ALUSrc_o;
output	[2-1:0]	RegDst_o, MemtoReg_o;
output			Jump_o, Branch_o, BranchType_o, MemWrite_o, MemRead_o;
 
//Internal Signals
wire	[3-1:0] ALUOp_o;
wire			ALUSrc_o;
wire		    RegWrite_o;
wire	[2-1:0]	RegDst_o, MemtoReg_o;
wire			Jump_o, Branch_o, BranchType_o, MemWrite_o, MemRead_o;

//Main function
/*your code here*/

reg	[3-1:0] ALUOp_o1;
reg			ALUSrc_o1;
reg		    RegWrite_o1;
reg	[2-1:0]	RegDst_o1, MemtoReg_o1;
reg			Jump_o1, Branch_o1, BranchType_o1, MemWrite_o1, MemRead_o1;
/*
assign RegWrite_o = (instr_op_i == 6'b000000) ? 1'b1 : (instr_op_i == 6'b001000) ? 1'b1 : (instr_op_i == 6'b100001) ? 1'b1 :
(instr_op_i == 6'b100011) ? 1'b0 : (instr_op_i == 6'b111011) ? 1'b0 : (instr_op_i == 6'100101) ? 1'b0 : 1'b0;

assign ALUOp_o = (instr_op_i == 6'b000000) ? 3'b010 : (instr_op_i == 6'b001000) ? 3'b011 : (instr_op_i == 6'b100001) ? 3'b000 :
(instr_op_i == 6'b100011) ? 3'b000 : (instr_op_i == 6'b111011) ? 3'b001 : (instr_op_i == 6'100101) ? 3'b110 : 3'b000;

assign ALUSrc_o = (instr_op_i == 6'b000000) ? 1'b0 : (instr_op_i == 6'b001000) ? 1'b1 : (instr_op_i == 6'b100001) ? 1'b1 :
(instr_op_i == 6'b100011) ? 1'b1 : (instr_op_i == 6'b111011) ? 1'b0 : (instr_op_i == 6'100101) ? 1'b0 : 1'b0;

assign RegDst_o =  (instr_op_i == 6'b001000) || (instr_op_i == 6'b100001) ? 2'b00 : 2'b01; 

assign Jump_o = (instr_op_i == 6'100101) ? 1'b1 : 1'b0; 

assign Branch_o = (instr_op_i == 6'b111011) || (instr_op_i == 6'100101) ? 1'b1 : 1'b0;

assign BranchType_o = (instr_op_i == 6'b000000) ?  : (instr_op_i == 6'b001000) ?  : (instr_op_i == 6'b100001) ?  :
(instr_op_i == 6'b100011) ?  : (instr_op_i == 6'b111011) ?  : (instr_op_i == 6'100101) ?  : 

*/




always @(*) begin
    case (instr_op_i)
        6'b000000:         begin     // R type
            RegWrite_o1 = 1'b1;
            ALUOp_o1 = 3'b010;
            ALUSrc_o1 = 1'b0;
            RegDst_o1 = 2'b01;
            Jump_o1 = 1'b0;
            Branch_o1 = 1'b0;
            BranchType_o1 = 1'b0;
            MemWrite_o1 = 1'b0;
            MemRead_o1 = 1'b0;
            MemtoReg_o1 = 2'b00;
            end
        6'b001000:          begin    // addi
            RegWrite_o1 = 1'b1;
            ALUOp_o1 = 3'b011;
            ALUSrc_o1 = 1'b1;
            RegDst_o1 = 2'b00;
            Jump_o1 = 1'b0;
            Branch_o1 = 1'b0;
            BranchType_o1 = 1'b0;
            MemWrite_o1 = 1'b0;
            MemRead_o1 = 1'b0;
            MemtoReg_o1 = 2'b00;
            end
        6'b100001:        begin      // lw
            RegWrite_o1 = 1'b1;
            ALUOp_o1 = 3'b000;
            ALUSrc_o1 = 1'b1;
            RegDst_o1 = 2'b00;
            Jump_o1 = 1'b0;
            Branch_o1 = 1'b0;
            BranchType_o1 = 1'b0;
            MemWrite_o1 = 1'b0;
            MemRead_o1 = 1'b1;
            MemtoReg_o1 = 2'b01;
            end
        6'b100011:        begin     // sw
            RegWrite_o1 = 1'b0;
            ALUOp_o1 = 3'b000;
            ALUSrc_o1 = 1'b1;
            RegDst_o1 = 2'b01;
            Jump_o1 = 1'b0;
            Branch_o1 = 1'b0;
            BranchType_o1 = 1'b0;
            MemWrite_o1 = 1'b1;
            MemRead_o1 = 1'b0;
            MemtoReg_o1 = 2'b00;
            end
        6'b111011:        begin      // beq
            RegWrite_o1 = 1'b0;
            ALUOp_o1 = 3'b001;
            ALUSrc_o1 = 1'b0;
            RegDst_o1 = 2'b01;
            Jump_o1 = 1'b0;
            Branch_o1 = 1'b1;
            BranchType_o1 = 1'b1;
            MemWrite_o1 = 1'b0;
            MemRead_o1 = 1'b0;
            MemtoReg_o1 = 2'b00;
            end
        6'b100101:        begin      // bne
            RegWrite_o1 = 1'b0;
            ALUOp_o1 = 3'b110;
            ALUSrc_o1 = 1'b0;
            RegDst_o1 = 2'b01;
            Jump_o1 = 1'b0;
            Branch_o1 = 1'b1;
            BranchType_o1 = 1'b0;
            MemWrite_o1 = 1'b0;
            MemRead_o1 = 1'b0;
            MemtoReg_o1 = 2'b00;
            end
        6'b100010:          begin    // j
            RegWrite_o1 = 1'b0;
            ALUOp_o1 = 3'b000;
            ALUSrc_o1 = 1'b0;
            RegDst_o1 = 2'b01;
            Jump_o1 = 1'b1;
            Branch_o1 = 1'b0;
            BranchType_o1 = 1'b0;
            MemWrite_o1 = 1'b0;
            MemRead_o1 = 1'b0;
            MemtoReg_o1 = 2'b00;
            end
        //default:begin
		//	$display("ERROR: invalid function code!!\nStop simulation");
		//	$display("%b",instr_op_i);
		//	$stop;
		//end
    endcase
end

assign RegWrite_o = RegWrite_o1;
assign ALUOp_o = ALUOp_o1;
assign ALUSrc_o = ALUSrc_o1;
assign RegDst_o = RegDst_o1;
assign Jump_o = Jump_o1;
assign Branch_o = Branch_o1;
assign BranchType_o = BranchType_o1;
assign MemWrite_o = MemWrite_o1;
assign MemRead_o = MemRead_o1;
assign MemtoReg_o = MemtoReg_o1;



endmodule
   