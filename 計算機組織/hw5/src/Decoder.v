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
wire			RegWrite_o;
wire	[2-1:0]	RegDst_o, MemtoReg_o;
wire			Jump_o, Branch_o, BranchType_o, MemWrite_o, MemRead_o;

//Main function
/*your code here*/

reg RegWrite_o1;
reg [3-1:0] ALUOp_o1;
reg ALUSrc_o1;
reg [2-1:0]	RegDst_o1, MemtoReg_o1;
reg Jump_o1, Branch_o1, BranchType_o1, MemWrite_o1, MemRead_o1;

always@(instr_op_i) begin
	case(instr_op_i)
		6'b000000: // R-Type
		begin
			RegWrite_o1 = 1;
			ALUOp_o1 = 3'b010;
			ALUSrc_o1 = 0;
			RegDst_o1 = 2'b01;
			Jump_o1 = 0;
			Branch_o1 = 0;
			BranchType_o1 = 0;
			MemWrite_o1 = 0;
			MemRead_o1 = 0;
			MemtoReg_o1 = 2'b00;
		end
		6'b111011: //beq
		begin
			RegWrite_o1 = 0;
			ALUOp_o1 = 3'b001;
			ALUSrc_o1 = 0;
			Jump_o1 = 0;
			Branch_o1 = 1;
			BranchType_o1 = 0;
			MemWrite_o1 = 0;
			MemRead_o1 = 0;
			
		end
		6'b100101: //bne
		begin
			RegWrite_o1 = 0;
			ALUOp_o1 = 3'b110;
			ALUSrc_o1 = 0;
			Jump_o1 = 0;
			Branch_o1 = 1;
			BranchType_o1 = 1;
			MemWrite_o1 = 0;
			MemRead_o1 = 0;
		end
		6'b011001: //blt
		begin
			RegWrite_o1 = 0;
			ALUSrc_o1 = 0;
			ALUOp_o1 = 3'b100;
			Jump_o1 = 0;
			Branch_o1 = 1;
			BranchType_o1 = 0;
			MemWrite_o1 = 0;
			MemRead_o1 = 0;
		end
		6'b001100: //bnez
		begin
			RegWrite_o1 = 0;
			ALUSrc_o1 = 0;
			ALUOp_o1 = 3'b101;
			Jump_o1 = 0;
			Branch_o1 = 1;
			BranchType_o1 = 1;
			MemWrite_o1 = 0;
			MemRead_o1 = 0;
		end
		6'b001110: //bgez
		begin
			RegWrite_o1 = 0;
			ALUSrc_o1 = 0;
			ALUOp_o1 = 3'b111;
			Jump_o1 = 0;
			Branch_o1 = 1;
			BranchType_o1 = 0;
			MemWrite_o1 = 0;
			MemRead_o1 = 0;
		end
		6'b100011: //sw
		begin
			RegWrite_o1 = 0;
			ALUOp_o1 = 3'b000;
			ALUSrc_o1 = 1;
			Jump_o1 = 0;
			Branch_o1 = 0;
			BranchType_o1 = 0;
			MemWrite_o1 = 1;
			MemRead_o1 = 0;
		end
		6'b100010: //jump
		begin
			RegWrite_o1 = 0;
			Jump_o1 = 1;
			Branch_o1 = 0;
			BranchType_o1 = 0;
			MemRead_o1 = 0;
			MemWrite_o1 = 0;
		end
		6'b000011: //jal
		begin
			RegWrite_o1 = 1;
			Jump_o1 = 1;
			RegDst_o1 = 2'b10;
			Branch_o1 = 0;
			BranchType_o1 = 0;
			MemtoReg_o1 = 2'b10;
			MemWrite_o1 = 0;
			MemRead_o1 = 0;
		end
		6'b100001: //lw
		begin
			RegWrite_o1 = 1;
			ALUOp_o1 = 3'b000;
			ALUSrc_o1 = 1;
			RegDst_o1 = 2'b00;
			Jump_o1 = 0;
			Branch_o1 = 0;
			BranchType_o1 = 0;
			MemWrite_o1 = 0;
			MemRead_o1 = 1;
			MemtoReg_o1 = 2'b01;
		end
		6'b001000: //addi
		begin
			RegWrite_o1 = 1;
			ALUOp_o1 = 3'b011;
			ALUSrc_o1 = 1;
			RegDst_o1 = 2'b00;
			Jump_o1 = 0;
			Branch_o1 = 0;
			BranchType_o1 = 0;
			MemWrite_o1 = 0;
			MemRead_o1 = 0;
			MemtoReg_o1 = 2'b00;
		end
		default:
		begin
			Jump_o1 = 0;
			Branch_o1 = 0;
			BranchType_o1 = 0;
		end

	endcase
	end

assign  RegWrite_o = RegWrite_o1;
assign  ALUOp_o = ALUOp_o1;
assign  ALUSrc_o = ALUSrc_o1;
assign  RegDst_o = RegDst_o1;
assign  Jump_o = Jump_o1;
assign  Branch_o = Branch_o1;
assign	BranchType_o = BranchType_o1;
assign	MemWrite_o = MemWrite_o1;
assign	MemRead_o = MemRead_o1;
assign	MemtoReg_o = MemtoReg_o1;

endmodule
   