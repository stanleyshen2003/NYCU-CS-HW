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
reg			 zero1;
reg			 overflow1;
reg	[32-1:0] result1;




always @(*) begin
    if(aluSrc1 == aluSrc2) begin
        zero1 = 1;
    end
    else begin
        zero1 = 0;
    end
    case(ALU_operation_i)
        4'b0010: begin
            result1 = aluSrc1 + aluSrc2;
            if((aluSrc1>0 && aluSrc2>0 && result1<0) || (aluSrc1<0 && aluSrc2<0 && result1>0)) begin
                overflow1 = 1;
            end
            else begin
                overflow1 = 0;
            end
        end
        4'b0110:   begin
            result1 = aluSrc1 - aluSrc2;
            if((aluSrc1>0 && result1<0) || (aluSrc1<0 && aluSrc2>0)) begin
                overflow1 = 1;
            end
            else begin
                overflow1 = 0;
            end
        end
        4'b0001:  begin
            result1 = aluSrc1 & aluSrc2;
            overflow1 = 0;
        end
        4'b0000:  begin
            result1 = aluSrc1 | aluSrc2;
            overflow1 = 0;
        end
        4'b1101:  begin
            result1 = ~(aluSrc1 | aluSrc2);
            overflow1 = 0;
        end
        4'b0111:   begin                           // overflow1 ??
            result1 = $signed(aluSrc1) < $signed(aluSrc2) ? 32'h00000001:32'h00000000;
            overflow1 = 0;
        end
    endcase


end
assign result = result1;
assign zero = zero1;
assign overflow = overflow1;



endmodule
