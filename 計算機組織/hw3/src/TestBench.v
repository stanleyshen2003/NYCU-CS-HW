`timescale 1ns/1ps
`define CYCLE_TIME 10			
`define END_COUNT 25

module TestBench;

//Internal Signals
reg         CLK;
reg         RST;
integer     count;
integer     handle;
integer     end_count;
//Top module
Simple_Single_CPU cpu(
        .clk_i(CLK),
		.rst_n(RST)
		);

//PARAMETER
//OP code definition
parameter TOTAL_INS  = 5;
reg  [6-1:0] INSTRUCION [TOTAL_INS-1:0];
parameter [6-1:0] OP_RTYPE = 6'b000000;
parameter [6-1:0] OP_ADDI  = 6'b001000;
parameter [6-1:0] OP_BEQ   = 6'b000100;
parameter [6-1:0] OP_ORI   = 6'b001101;
parameter [6-1:0] OP_LUI   = 6'b001111;

parameter TOTAL_FUNC = 10;
reg  [6-1:0] FUNCTION [TOTAL_FUNC-1:0];
parameter [6-1:0] FUNC_ADD = 6'b010010;
parameter [6-1:0] FUNC_SUB = 6'b010000;
parameter [6-1:0] FUNC_AND = 6'b010100;
parameter [6-1:0] FUNC_OR  = 6'b010110;
parameter [6-1:0] FUNC_SLT = 6'b100000;
parameter [6-1:0] FUNC_SLLV= 6'b000110;
parameter [6-1:0] FUNC_SLL = 6'b000000;
parameter [6-1:0] FUNC_SRLV= 6'b000100;
parameter [6-1:0] FUNC_SRL = 6'b000010;
parameter [6-1:0] FUNC_NOR = 6'b010101;

//Other register declaration
reg [31:0] instruction;
reg signed [31:0] register_file[31:0];
reg [31:0] pc;
reg [4:0]  rs,rt,rd;
integer i;

always #(`CYCLE_TIME/2) CLK = ~CLK;	

initial begin
// update instruction and function field
INSTRUCION[0] = OP_RTYPE;
INSTRUCION[1] = OP_ADDI;
INSTRUCION[2] = OP_LUI;
INSTRUCION[3] = OP_BEQ;
INSTRUCION[4] = OP_ORI;


FUNCTION[0] = FUNC_ADD;
FUNCTION[1] = FUNC_SUB;
FUNCTION[2] = FUNC_AND;
FUNCTION[3] = FUNC_OR;
FUNCTION[4] = FUNC_SLT;
FUNCTION[5] = FUNC_NOR;
FUNCTION[6] = FUNC_SLL;
FUNCTION[7] = FUNC_SRL;

for(i=0;i<32;i=i+1)begin
	register_file[i] = 32'd0;
end
end

initial  begin
	$readmemb("CO_P3_test_data1.txt", cpu.IM.Instr_Mem);  //Read instruction from "CO_P3_test_data1.txt" 
    handle = $fopen("CO_P3_result.txt");
	
	CLK = 0;
    RST = 0;
	count = 0;
    end_count=25;
	instruction = 32'd0;
    @(negedge CLK);
	RST = 1;
	pc = 32'd0;
	
	for(;count != `END_COUNT;)begin
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
						register_file[rd] = (register_file[rs] < register_file[rt]) ?(32'd1):(32'd0) ;
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
					FUNC_NOR:begin
						register_file[rd] = ~register_file[rs] & ~register_file[rt];
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
			OP_ORI:begin
				rs = instruction[25:21];
				rt = instruction[20:16];
				register_file[rt] = register_file[rs]  |  {{16'd0},{instruction[15:0]}};
			end
			OP_LUI:begin
				rs = instruction[25:21];
				rt = instruction[20:16];
				register_file[rt] = $signed(register_file[rs]) + $signed({{instruction[15:0]},{16'h0000}});
			end
			default:begin
				$display("ERROR: invalid op code!!\nStop simulation");
				#(`CYCLE_TIME*1);
				$stop;
			end
		endcase
	
	
		
		@(negedge CLK);
		// after the register file of design is updated
		// compare the register file that are 	
		if(instruction[31:26] == OP_BEQ)begin
			if(cpu.PC.pc_out_o !== pc)begin
				$display("ERROR: BEQ instruction fail");
				$display("The correct pc address is %d",pc);
				$display("Your pc address is %d",cpu.PC.pc_out_o);
				$stop;
			end
		end
		else begin
			if(cpu.PC.pc_out_o !== pc)begin
				$display("ERROR: Your next PC points to wrong address");
				$display("The correct pc address is %d",pc);
				$display("Your pc address is %d",cpu.PC.pc_out_o);
				$stop;
			end
		end
		
		// Check the register file
		// It should be the same with the register file in the design
		for(i=0; i<31; i=i+1)begin
			if(cpu.RF.Reg_File[i] !== register_file[i])begin
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
							FUNC_NOR:begin
								$display("ERROR: NOR  instruction fail");
							end
						endcase
					end
					OP_ADDI:begin
						$display("ERROR: ADDI instruction fail");
					end
					OP_BEQ:begin
						$display("ERROR: BEQ  instruction fail");
					end
					OP_ORI:begin
						$display("ERROR: ORI  instruction fail");
					end
					OP_LUI:begin
						$display("ERROR: LUI  instruction fail");
					end
				endcase
				$display("Register %d contains wrong answer",i);
				$display("The correct value is %d ",register_file[i]);
				$display("Your wrong value is %d ",cpu.RF.Reg_File[i]);
				$stop;
			end
		end
		if(cpu.IM.Instr_Mem[ pc >>2 ] == 32'd0)begin
			count = `END_COUNT;    
			 #(`CYCLE_TIME*2);

		end
		else begin
			count = count + 1;
		end
	end
	$display("============================================");
	$display("   Congratulation. You pass  TA's pattern   ");
	$display("============================================");
    	
			$fdisplay(handle, "r0=%d, r1=%d, r2=%d, r3=%d, r4=%d, r5=%d, r6=%d, r7=%d, r8=%d, r9=%d, r10=%d, r11=%d, r12=%d, r13=%d, r14=%d ",
	          cpu.RF.Reg_File[0], cpu.RF.Reg_File[1], cpu.RF.Reg_File[2], cpu.RF.Reg_File[3], cpu.RF.Reg_File[4], 
			  cpu.RF.Reg_File[5], cpu.RF.Reg_File[6], cpu.RF.Reg_File[7], cpu.RF.Reg_File[8], cpu.RF.Reg_File[9], 
			  cpu.RF.Reg_File[10],cpu.RF.Reg_File[11], cpu.RF.Reg_File[12], cpu.RF.Reg_File[13], cpu.RF.Reg_File[14]
			);
    $fclose(handle); $stop;
end

  
endmodule
