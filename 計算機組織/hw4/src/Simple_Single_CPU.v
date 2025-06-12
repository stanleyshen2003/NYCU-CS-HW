module Simple_Single_CPU( clk_i, rst_n );

//I/O port
input         clk_i;
input         rst_n;

//Internal Signles
wire [32-1:0] instr, PC_i, PC_o, ReadData1, ReadData2, WriteData;
wire [32-1:0] signextend, zerofilled, ALUinput2, ALUResult, ShifterResult;
wire [5-1:0] WriteReg_addr, Shifter_shamt;
wire [4-1:0] ALU_operation;
wire [3-1:0] ALUOP;
wire [2-1:0] FURslt;
wire [2-1:0] RegDst, MemtoReg;
wire RegWrite, ALUSrc, zero, overflow;
wire Jump, Branch, BranchType, MemWrite, MemRead;
wire [32-1:0] PC_add1, PC_add2, PC_no_jump, PC_t, Mux3_result, DM_ReadData;
wire Jr;
assign Jr = ((instr[31:26] == 6'b000000) && (instr[20:0] == 21'd8)) ? 1 : 0;
//modules
/*your code here*/
Program_Counter PC( clk_i, rst_n, PC_i, PC_o );
wire [31:0] four;
assign four = 32'b00000000000000000000000000000100;
Adder adder(PC_o, four, PC_add1);
Decoder decoder( instr[31:26], RegWrite, ALUOP, ALUSrc, RegDst, Jump, Branch, BranchType, MemWrite, MemRead, MemtoReg); 
Mux2to1 #(.size(5)) mux1(instr[20:16], instr[15:11], RegDst[0], WriteReg_addr);                                                     //
Instr_Memory IM(PC_o, instr);

assign PC_t[27:2] = instr[25:0];
assign PC_t[1:0] = 2'b00;
assign PC_t[31:28] = PC_add1[31:28];
/*
always @(signextend) begin
    $display("signextend = %b", signextend);
end
*/
wire [31:0] tempsignextend;
assign tempsignextend[31:2] = signextend[29:0];
assign tempsignextend[1:0] = 2'b00;
//assign PC_add2 = tempsignextend + PC_add1;
Adder add2(PC_add1, tempsignextend, PC_add2);
wire PCSrc, choosetype;
assign PCSrc = Branch & choosetype;
Mux2to1 #(.size(32)) mux2(PC_add1, PC_add2, PCSrc, PC_no_jump);                                                                       //
Mux2to1 #(.size(32)) mux3(PC_no_jump, PC_t , Jump, PC_i);                                                                             //

Reg_File RF( clk_i, rst_n, instr[25:21], instr[20:16], WriteReg_addr, WriteData, RegWrite, ReadData1, ReadData2 );
Sign_Extend signext(instr[15:0], signextend[31:0]);
Zero_Filled zerofillll(instr[15:0], zerofilled);

Mux2to1 #(.size(32)) mux4(ReadData2, signextend[31:0], ALUSrc, ALUinput2); 
/*                                                                //
always @ (ALUinput2, ALUSrc) begin
    $display("mux4 : ALUinput2 = %b, ALUSrc = %b, signextend = %b", ALUinput2, ALUSrc, signextend);
end
*/
/*
always @(Jump) begin
    $display("Jump = %b, PC_o = %b", Jump, PC_o);

end
*/


ALU alu( ReadData1, ALUinput2, ALU_operation, ALUResult, zero, overflow );

ALU_Ctrl aluControl( instr[5:0], ALUOP, ALU_operation, FURslt );                                                        

wire temp;
assign temp = ~instr[1];
Shifter shifter( ShifterResult, temp, instr[10:6], ALUinput2 );

wire notzero;
assign notzero = ~zero;
Mux2to1 #(.size(1)) mux5(notzero, zero, BranchType, choosetype);                                                                    //


Mux3to1 #(.size(32)) mux6( ALUResult, ShifterResult, zerofilled, FURslt, Mux3_result );                                              //

Data_Memory DM( clk_i, Mux3_result, ReadData2, MemRead, MemWrite, DM_ReadData);

Mux2to1 #(.size(32)) mux7(Mux3_result, DM_ReadData, MemtoReg[0], WriteData);                                                         //

endmodule



