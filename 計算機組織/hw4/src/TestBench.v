`timescale 1ns/1ps
`define CYCLE_TIME 10			
`define END_COUNT 600

`include "Adder.v"
`include "ALU.v"
`include "ALU_Ctrl.v"
`include "Data_Memory.v"
`include "Decoder.v"
`include "Instr_Memory.v"
`include "Mux2to1.v"
`include "Mux3to1.v"
`include "Program_Counter.v"
`include "Reg_File.v"
`include "Shifter.v"
`include "Sign_Extend.v"
`include "Simple_Single_CPU.v"
`include "Zero_Filled.v"

module TestBench;

//Internal Signals
reg         CLK;
reg         RST;
integer     count;
integer     handle;
integer     end_count;
//Greate tested modle  
Simple_Single_CPU cpu(
        .clk_i(CLK),
		.rst_n(RST)
		);

//PARAMETER
//OP code definition
parameter TOTAL_INS  = 11;
reg  [6-1:0] INSTRUCION [TOTAL_INS-1:0];
parameter [6-1:0] OP_RTYPE = 6'b000000;
parameter [6-1:0] OP_ADDI  = 6'b001000;
parameter [6-1:0] OP_BEQ   = 6'b111011;
parameter [6-1:0] OP_LUI   = 6'b001111;
parameter [6-1:0] OP_LW	   = 6'b100001;
parameter [6-1:0] OP_SW    = 6'b100011;
parameter [6-1:0] OP_BNE   = 6'b100101;
parameter [6-1:0] OP_JUMP  = 6'b100010;
parameter [6-1:0] OP_JAL   = 6'b000011;
parameter [6-1:0] OP_BLT   = 6'b011001;
parameter [6-1:0] OP_BNEZ  = 6'b001100;
parameter [6-1:0] OP_BGEZ  = 6'b001110;

parameter TOTAL_FUNC = 11;
reg  [6-1:0] FUNCTION [TOTAL_FUNC-1:0];
parameter [6-1:0] FUNC_ADD = 6'b010010;
parameter [6-1:0] FUNC_SUB = 6'b010000;
parameter [6-1:0] FUNC_AND = 6'b010100;
parameter [6-1:0] FUNC_OR  = 6'b010110;
parameter [6-1:0] FUNC_NOT = 6'b101111;
parameter [6-1:0] FUNC_SLT = 6'b100000;
parameter [6-1:0] FUNC_SLLV= 6'b000110;
parameter [6-1:0] FUNC_SLL = 6'b000000;
parameter [6-1:0] FUNC_SRLV= 6'b000100;
parameter [6-1:0] FUNC_SRL = 6'b000010;
parameter [6-1:0] FUNC_JR  = 6'b001000;

//Other register declaration
reg [31:0] instruction;
reg [31:0] register_file[31:0];
reg [7:0]  mem_file [0:127];
reg [31:0] pc;
reg [4:0]  rs,rt,rd;
reg [31:0] addr,data;
integer i;
integer j;
reg [3:0] k;
integer mode;
integer score;
integer initail_error;
integer error_record;

wire [31:0] memory[0:31];
assign  memory[0] = {mem_file[3], mem_file[2], mem_file[1], mem_file[0]};
assign  memory[1] = {mem_file[7], mem_file[6], mem_file[5], mem_file[4]};
assign  memory[2] = {mem_file[11], mem_file[10], mem_file[9], mem_file[8]};
assign  memory[3] = {mem_file[15], mem_file[14], mem_file[13], mem_file[12]};
assign  memory[4] = {mem_file[19], mem_file[18], mem_file[17], mem_file[16]};
assign  memory[5] = {mem_file[23], mem_file[22], mem_file[21], mem_file[20]};
assign  memory[6] = {mem_file[27], mem_file[26], mem_file[25], mem_file[24]};
assign  memory[7] = {mem_file[31], mem_file[30], mem_file[29], mem_file[28]};
assign  memory[8] = {mem_file[35], mem_file[34], mem_file[33], mem_file[32]};
assign  memory[9] = {mem_file[39], mem_file[38], mem_file[37], mem_file[36]};
assign  memory[10] = {mem_file[43], mem_file[42], mem_file[41], mem_file[40]};
assign  memory[11] = {mem_file[47], mem_file[46], mem_file[45], mem_file[44]};
assign  memory[12] = {mem_file[51], mem_file[50], mem_file[49], mem_file[48]};
assign  memory[13] = {mem_file[55], mem_file[54], mem_file[53], mem_file[52]};
assign  memory[14] = {mem_file[59], mem_file[58], mem_file[57], mem_file[56]};
assign  memory[15] = {mem_file[63], mem_file[62], mem_file[61], mem_file[60]};
assign  memory[16] = {mem_file[67], mem_file[66], mem_file[65], mem_file[64]};
assign  memory[17] = {mem_file[71], mem_file[70], mem_file[69], mem_file[68]};
assign  memory[18] = {mem_file[75], mem_file[74], mem_file[73], mem_file[72]};
assign  memory[19] = {mem_file[79], mem_file[78], mem_file[77], mem_file[76]};
assign  memory[20] = {mem_file[83], mem_file[82], mem_file[81], mem_file[80]};
assign  memory[21] = {mem_file[87], mem_file[86], mem_file[85], mem_file[84]};
assign  memory[22] = {mem_file[91], mem_file[90], mem_file[89], mem_file[88]};
assign  memory[23] = {mem_file[95], mem_file[94], mem_file[93], mem_file[92]};
assign  memory[24] = {mem_file[99], mem_file[98], mem_file[97], mem_file[96]};
assign  memory[25] = {mem_file[103], mem_file[102], mem_file[101], mem_file[100]};
assign  memory[26] = {mem_file[107], mem_file[106], mem_file[105], mem_file[104]};
assign  memory[27] = {mem_file[111], mem_file[110], mem_file[109], mem_file[108]};
assign  memory[28] = {mem_file[115], mem_file[114], mem_file[113], mem_file[112]};
assign  memory[29] = {mem_file[119], mem_file[118], mem_file[117], mem_file[116]};
assign  memory[30] = {mem_file[123], mem_file[122], mem_file[121], mem_file[120]};
assign  memory[31] = {mem_file[127], mem_file[126], mem_file[125], mem_file[124]};

always #(`CYCLE_TIME/2) CLK = ~CLK;	

initial begin
// update instruction and function field
INSTRUCION[0] = OP_RTYPE;
INSTRUCION[1] = OP_ADDI;
INSTRUCION[2] = OP_BEQ;
INSTRUCION[2] = OP_BNE;

FUNCTION[0] = FUNC_ADD;
FUNCTION[1] = FUNC_SUB;
FUNCTION[2] = FUNC_AND;
FUNCTION[3] = FUNC_OR;
FUNCTION[4] = FUNC_SLT;
for(i=0;i<32;i=i+1)begin
	register_file[i] = 32'd0;
end
	register_file[29]= 32'd128; //stack pointer
for(i=0; i<128; i=i+1)begin
		mem_file[i] = 8'b0;
end

end

initial begin
	$readmemb("CO_P4_test_data1.txt", cpu.IM.Instr_Mem);		mode = 0;  // (A) basic
	//$readmemb("CO_P4_test_data2.txt", cpu.IM.Instr_Mem);		mode = 1;  // (B) advance set1 (Fibonacci)
	//$readmemb("CO_P4_test_data2_2.txt", cpu.IM.Instr_Mem);	mode = 2;  // (C) advance set2
    handle = $fopen("CO_P4_result.txt");
	
	CLK = 0;
    RST = 0;
	count = 0;
	instruction = 32'd0;
    @(negedge CLK);
	RST = 1;
	pc = cpu.PC.pc_out_o;
	
	while(count != `END_COUNT)begin
		//instruction = cpu.IM.Instr_Mem[ cpu.PC.pc_out_o>>2 ];
		instruction = cpu.IM.Instr_Mem[ pc>>2 ];
		pc = pc + 32'd4;
		
		// check wether
		
		case(instruction[31:26])
			OP_RTYPE:begin
				rs = instruction[25:21];
				rt = instruction[20:16];
				rd = instruction[15:11];
				case(instruction[5:0])
					FUNC_ADD:begin
						register_file[rd] = $signed(register_file[rs]) + $signed(register_file[rt]);
					end
					FUNC_SUB:begin
						register_file[rd] = $signed(register_file[rs]) - $signed(register_file[rt]);
					end
					FUNC_AND:begin
						register_file[rd] = register_file[rs] & register_file[rt] ;
					end
					FUNC_OR:begin
						register_file[rd] = register_file[rs] | register_file[rt] ;
					end
					FUNC_SLT:begin
						register_file[rd] = ($signed(register_file[rs]) < $signed(register_file[rt])) ?(32'd1):(32'd0) ;
					end
					FUNC_SLLV:begin
						register_file[rd] = register_file[rt] << register_file[rs];
					end
					FUNC_SLL:begin
						register_file[rd] = register_file[rt] << instruction[10:6];
					end
					FUNC_SRLV:begin
						register_file[rd] = register_file[rt] >> register_file[rs];
					end
					FUNC_SRL:begin
						register_file[rd] = register_file[rt] >> instruction[10:6];
					end
					FUNC_NOT:begin
						register_file[rd] = ~register_file[rs];
					end
					FUNC_JR:begin
						pc = register_file[rs];
					end
					default:begin
						$display("ERROR: invalid function code!!\nStop simulation");
						#(`CYCLE_TIME*1);
						$stop;
					end
				endcase
			end
			OP_ADDI:begin
				rs = instruction[25:21];
				rt = instruction[20:16];
				register_file[rt] = $signed(register_file[rs]) + $signed({{16{instruction[15]}},{instruction[15:0]}});
			end
			OP_BEQ:begin
				rs = instruction[25:21];
				rt = instruction[20:16];
				if(register_file[rt] == register_file[rs])begin
					pc = pc + $signed({{14{instruction[15]}},{instruction[15:0]},{2'd0}}) ;
				end
			end
			OP_LW:begin
				rs = instruction[25:21];
				rt = instruction[20:16];
				addr = $signed(register_file[rs]) + $signed({{16{instruction[15]}},{instruction[15:0]}});
				register_file[rt] = {mem_file[addr+3], mem_file[addr+2], mem_file[addr+1], mem_file[addr]};
			end
			OP_SW:begin
				rs = instruction[25:21];
				rt = instruction[20:16];
				addr = $signed(register_file[rs]) + $signed({{16{instruction[15]}},{instruction[15:0]}});
				data = register_file[rt];
				mem_file[addr+3] <= data[31:24];
				mem_file[addr+2] <= data[23:16];
				mem_file[addr+1] <= data[15:8];
				mem_file[addr]   <= data[7:0];
			end
			OP_JUMP:begin
				pc = {{pc[31:28]},{instruction[25:0]},{2'b00}};
			end
			OP_BNE:begin
				rs = instruction[25:21];
				rt = instruction[20:16];
				if(register_file[rt] != register_file[rs])begin
					pc = pc + $signed({{14{instruction[15]}},{instruction[15:0]},{2'd0}}) ;
				end
			end
			OP_LUI:begin
				rs = instruction[25:21];
				rt = instruction[20:16];
				register_file[rt] = $signed(register_file[rs]) + $signed({{instruction[15:0]},{16'h0000}});
			end
			OP_JAL:begin
				register_file[31] = pc;
				pc = {{pc[31:28]},{instruction[25:0]},{2'b00}};
			end
			OP_BLT:begin
				rs = instruction[25:21];
				rt = instruction[20:16];
				if($signed(register_file[rs]) < $signed(register_file[rt]))begin
					pc = pc + $signed({{14{instruction[15]}},{instruction[15:0]},{2'd0}}) ;
				end
			end
			OP_BNEZ:begin
				rs = instruction[25:21];
				if($signed(register_file[rs]) !=0)begin
					pc = pc + $signed({{14{instruction[15]}},{instruction[15:0]},{2'd0}}) ;
				end
			end
			OP_BGEZ:begin
				rs = instruction[25:21];
				if($signed(register_file[rs]) >=0)begin
					pc = pc + $signed({{14{instruction[15]}},{instruction[15:0]},{2'd0}}) ;
				end
			end
			default:begin
				$display("ERROR: invalid op code!!\nStop simulation");
				#(`CYCLE_TIME*1);
				$stop;
			end
		endcase
	
	
		
		@(negedge CLK);
		// after the pc of design is updated
		// compare the pc value that are 	
		/*if(cpu.PC.pc_out_o !== pc)begin
			case(instruction[31:26])
				OP_RTYPE:begin
					if (instruction[5:0]==FUNC_JR)begin
						$display("ERROR: JR  instruction fail");
					end
				end
				OP_BEQ:begin
					$display("ERROR: BEQ  instruction fail");
				end	
				OP_JUMP:begin
					$display("ERROR: JUMP  instruction fail");
				end
				OP_BNE:begin
					$display("ERROR: BNE  instruction fail");
				end
				OP_JAL:begin
					$display("ERROR: JAL  instruction fail");
				end
				OP_BLT:begin
					$display("ERROR: BLT  instruction fail");
				end
				OP_BGEZ:begin
					$display("ERROR: BGEZ  instruction fail");
				end
				default:begin
					//$display("ERROR: Your next PC points to wrong address");
				end
			endcase
			//$display("The correct pc address is %d",pc);
			//$display("Your pc address is %d",cpu.PC.pc_out_o);
			//$stop;
		end
		
		// Check the register & memory file
		// It should be the same with the register file in the design
		for(i=0; i<32; i=i+1)begin
			if(cpu.RF.Reg_File[i] !== register_file[i] || cpu.DM.memory[i] !== memory[i])begin
				case(instruction[31:26])
					OP_RTYPE:begin
						case(instruction[5:0])
							FUNC_ADD:begin
								$display("ERROR: ADD instruction fail");
							end
							FUNC_SUB:begin
								$display("ERROR: SUB instruction fail");
							end
							FUNC_AND:begin
								$display("ERROR: AND instruction fail");
							end
							FUNC_OR:begin
								$display("ERROR: OR  instruction fail");
							end
							FUNC_SLT:begin
								$display("ERROR: SLT instruction fail");
							end
							FUNC_SLLV:begin
								$display("ERROR: SLLV instruction fail");
							end
							FUNC_SLL:begin
								$display("ERROR: SLL  instruction fail");
							end
							FUNC_SRLV:begin
								$display("ERROR: SRLV instruction fail");
							end
							FUNC_SRL:begin
								$display("ERROR: SRL  instruction fail");
							end
							FUNC_NOT:begin
								$display("ERROR: NOT  instruction fail");
							end
							FUNC_JR:begin
								$display("ERROR: JR  instruction fail");
							end
						endcase
					end
					OP_ADDI:begin
						$display("ERROR: ADDI instruction fail");
					end
					OP_BEQ:begin
						$display("ERROR: BEQ  instruction fail");
					end
					OP_LW:begin
						$display("ERROR: LW  instruction fail");
					end
					OP_SW:begin
						$display("ERROR: SW  instruction fail");
					end
					OP_JUMP:begin
						$display("ERROR: JUMP  instruction fail");
					end
					OP_BNE:begin
						$display("ERROR: BNE  instruction fail");
					end
					OP_LUI:begin
						$display("ERROR: LUI  instruction fail");
					end
					OP_JAL:begin
						$display("ERROR: JAL  instruction fail");
					end
					OP_BLT:begin
						$display("ERROR: BLT  instruction fail");
					end
					OP_BGEZ:begin
						$display("ERROR: BGEZ  instruction fail");
					end
				endcase
				if(cpu.RF.Reg_File[i] !== register_file[i])begin
					$display("Register %d contains wrong answer",i);
					$display("The correct value is %d ",register_file[i]);
					$display("Your wrong value is %d ",cpu.RF.Reg_File[i]);
				end
				if(cpu.DM.memory[i] !== memory[i])begin
					$display("Memory %d contains wrong answer",i);
					$display("The correct value is %d ",memory[i]);
					$display("Your wrong value is %d ",cpu.DM.memory[i]);
				end
				//$stop;
			end
		end*/
		if(cpu.IM.Instr_Mem[ pc>>2 ] == 32'd0)begin
			count = `END_COUNT;    
			 #(`CYCLE_TIME*2);

		end
		else begin
			count = count + 1;
		end
	end

	score = 0;
	error_record=0;
	case(mode)
		0:begin
			initail_error = 0;
			for(i=0; i<32; i=i+1)begin
				if(i<16 && cpu.RF.Reg_File[i] == register_file[i])begin
					score = score + 2;
				end
				if(i==29 && cpu.RF.Reg_File[i] == register_file[i])begin
					score = score + 3;
				end
				if(i>=16 && i!==29 && initail_error==0 && cpu.RF.Reg_File[i] !== register_file[i])begin
					initail_error = 1;
				end
				if(i<10 && cpu.DM.memory[i] == memory[i])begin
					score = score + 4;
				end
				if(i>=10 && initail_error==0 && cpu.DM.memory[i] !== memory[i])begin
					initail_error = 1;
				end
			end
			if(initail_error !== 0)begin
				$display("ERROR: initail error");
				error_record = 1;
				score = score - 5;
				if(score < 0)begin
					score = 0;
				end
			end
			if(cpu.RF.Reg_File[29] !== register_file[29])begin
				$display("ERROR: stack point error");
				error_record = 1;
			end
			for(i=0; i<16; i=i+1)begin
				j=i-6;
				if(cpu.RF.Reg_File[i] !== register_file[i])begin
					k=i;
					case(i)
						1:begin $display("ERROR: ADDI"); end
						2:begin $display("ERROR: ADDI"); end
						3:begin $display("ERROR: ADDI"); end
						4:begin $display("ERROR: ADDI"); end
						5:begin	$display("ERROR: SUB or JUMP");	end
						6:begin	$display("ERROR: LW");	end
						7:begin	$display("ERROR: LW");	end
						8:begin $display("ERROR: BEQ(r%d)",k); end
						9:begin $display("ERROR: BEQ(r%d)",k); end
						10:begin $display("ERROR: BEQ(r%d)",k); end
						11:begin $display("ERROR: BNE(r%d)",k);	end
						12:begin $display("ERROR: BNE(r%d)",k);	end
						13:begin $display("ERROR: BNE(r%d)",k);	end
						14:begin $display("ERROR: JUMP(r%d)",k); end
						15:begin $display("ERROR: JUMP(r%d)",k); end
					endcase
					error_record = 1;
				end
				if(j>=0 && cpu.DM.memory[j] !== memory[j])begin
					k=j;
					case(j)
						0:begin $display("ERROR: SW(m%d)",k); end
						1:begin $display("ERROR: SW(m%d)",k); end
						2:begin $display("ERROR: BEQ(m%d)",k); end
						3:begin $display("ERROR: BEQ(m%d)",k); end
						4:begin $display("ERROR: BEQ(m%d)",k); end
						5:begin	$display("ERROR: BNE(m%d)",k);	end
						6:begin	$display("ERROR: BNE(m%d)",k);	end
						7:begin	$display("ERROR: BNE(m%d)",k);	end
						8:begin $display("ERROR: JUMP(m%d)",k); end
						9:begin $display("ERROR: JUMP(m%d)",k); end
					endcase
					error_record = 1;
				end
			end
			
			$display("============================================");
			$display("(A) basic score: %d / 75",score);
			if(score !== 75)begin
				error_record=1;
			end
		end
		1:begin
			for(i=0; i<4; i=i+1)begin
				if(cpu.DM.memory[i] == memory[i])begin
					score = score + 3;
				end
			end

			if(cpu.DM.memory[4] == memory[4])begin
				score = score + 2;
			end
			if(cpu.DM.memory[5] == memory[5])begin
				score = score + 1;
			end
			if(cpu.RF.Reg_File[29] !== register_file[29])begin
				$display("ERROR: stack point error");
				score = score - 5;
				error_record = 1;
			end
			$display("============================================");
			$display("(B) advance set1: %d / 15",score);
			if(score !== 15)begin
				error_record=1;
			end
		end
		2:begin
			for(i=0; i<5; i=i+1)begin
				if(cpu.DM.memory[i] == memory[i])begin
					score = score + 2;
				end
			end
			if(cpu.DM.memory[0] !== memory[0])begin
				$display("ERROR: BLT");
				score = score - 2;
				error_record = 1;
			end
			/*if(cpu.DM.memory[1] !== memory[1])begin
				$display("ERROR: BNEZ");
				score = score - 2;
				error_record = 1;
			end*/
			if(cpu.DM.memory[2] !== memory[2] || cpu.DM.memory[3] !== memory[3])begin
				$display("ERROR: BGEZ");
				score = score - 2;
				error_record = 1;
			end

			$display("============================================");
			$display("(C) advance set2: %d / 10",score);
			if(score !== 10)begin
				error_record=1;
			end
		end
		3:begin
			for(i=0; i<5; i=i+1)begin
				//if(cpu.DM.memory[i] == memory[i])begin
				if(cpu.DM.memory[i] == i+1)begin
					score = score + 3;
				end
			end
			//$display("============================================");
			//$display("(D) bonus score: %d / 15",score);
			//$display("    Please check \"CO_P4_test_data3.txt\" for another 5 pts.");
		end
	endcase
	if(error_record == 0)begin
		$display("============================================");
		$display("   Congratulation. You pass  TA's pattern   ");
		//$display("============================================");
	end
	$display("============================================");
	
			$fdisplay(handle,"Register======================================================");
			$fdisplay(handle, 
"r0=%d, r1=%d, r2=%d, r3=%d,\n\
r4=%d, r5=%d, r6=%d, r7=%d,\n\
r8=%d, r9=%d, r10=%d, r11=%d,\n\
r12=%d, r13=%d, r14=%d, r15=%d,\n\
r16=%d, r17=%d, r18=%d, r19=%d,\n\
r20=%d, r21=%d, r22=%d, r23=%d,\n\
r24=%d, r25=%d, r26=%d, r27=%d,\n\
r28=%d, r29=%d, r30=%d, r31=%d,\n",
	          cpu.RF.Reg_File[0], cpu.RF.Reg_File[1], cpu.RF.Reg_File[2], cpu.RF.Reg_File[3], cpu.RF.Reg_File[4], 
			  cpu.RF.Reg_File[5], cpu.RF.Reg_File[6], cpu.RF.Reg_File[7], cpu.RF.Reg_File[8], cpu.RF.Reg_File[9], 
			  cpu.RF.Reg_File[10],cpu.RF.Reg_File[11], cpu.RF.Reg_File[12], cpu.RF.Reg_File[13], cpu.RF.Reg_File[14],
			  cpu.RF.Reg_File[15], cpu.RF.Reg_File[16], cpu.RF.Reg_File[17], cpu.RF.Reg_File[18], cpu.RF.Reg_File[19], 
			  cpu.RF.Reg_File[20], cpu.RF.Reg_File[21], cpu.RF.Reg_File[22], cpu.RF.Reg_File[23], cpu.RF.Reg_File[24], 
			  cpu.RF.Reg_File[25],cpu.RF.Reg_File[26], cpu.RF.Reg_File[27], cpu.RF.Reg_File[28], cpu.RF.Reg_File[29],
			  cpu.RF.Reg_File[30],cpu.RF.Reg_File[31]);
			$fdisplay(handle,"Memory========================================================");
			$fdisplay(handle, 
"m0=%d, m1=%d, m2=%d, m3=%d,\n\
m4=%d, m5=%d, m6=%d, m7=%d,\n\
m8=%d, m9=%d, m10=%d, m11=%d,\n\
m12=%d, m13=%d, m14=%d, m15=%d,\n\
m16=%d, m17=%d, m18=%d, m19=%d,\n\
m20=%d, m21=%d, m22=%d, m23=%d,\n\
m24=%d, m25=%d, m26=%d, m27=%d,\n\
m28=%d, m29=%d, m30=%d, m31=%d,\n",
	          cpu.DM.memory[0], cpu.DM.memory[1], cpu.DM.memory[2], cpu.DM.memory[3], cpu.DM.memory[4], 
			  cpu.DM.memory[5], cpu.DM.memory[6], cpu.DM.memory[7], cpu.DM.memory[8], cpu.DM.memory[9], 
			  cpu.DM.memory[10],cpu.DM.memory[11], cpu.DM.memory[12], cpu.DM.memory[13], cpu.DM.memory[14],
			  cpu.DM.memory[15], cpu.DM.memory[16], cpu.DM.memory[17], cpu.DM.memory[18], cpu.DM.memory[19], 
			  cpu.DM.memory[20], cpu.DM.memory[21], cpu.DM.memory[22], cpu.DM.memory[23], cpu.DM.memory[24], 
			  cpu.DM.memory[25],cpu.DM.memory[26], cpu.DM.memory[27], cpu.DM.memory[28], cpu.DM.memory[29],
			  cpu.DM.memory[30],cpu.DM.memory[31]);
    $fclose(handle); $stop;
end

  
endmodule
