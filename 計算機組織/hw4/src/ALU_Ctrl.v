module ALU_Ctrl( funct_i, ALUOp_i, ALU_operation_o, FURslt_o );

//I/O ports 
input      [6-1:0] funct_i;
input      [3-1:0] ALUOp_i;

output     [4-1:0] ALU_operation_o;  
output     [2-1:0] FURslt_o;
     
//Internal Signals
wire	    [4-1:0] ALU_operation_o;
wire		[2-1:0] FURslt_o;

//Main function
/*your code here*/

reg	    [4-1:0] ALU_operation_o1;
reg		[2-1:0] FURslt_o1;
always @(*) begin
    case(ALUOp_i)
        3'b000:     begin// lw sw 
            FURslt_o1 = 2'b00;
            ALU_operation_o1 = 4'b0010;
        end
        3'b001:     begin // beq
            FURslt_o1 = 2'b00;
            ALU_operation_o1 = 4'b0110;
        end
        3'b110:     begin // bne
            FURslt_o1 = 2'b00;
            ALU_operation_o1 = 4'b0110;
        end
        3'b011:     begin // addi
            FURslt_o1 = 2'b00;
            ALU_operation_o1 = 4'b0010;
        end
        3'b010:      begin
            if(funct_i == 6'b010010) begin          // ADD
                FURslt_o1 = 2'b00;
                ALU_operation_o1 = 4'b0010;
            end
            else if(funct_i == 6'b010000) begin     // SUB
                FURslt_o1 = 2'b00;
                ALU_operation_o1 = 4'b0110;
            end
            else if(funct_i == 6'b010100) begin     // AND
                FURslt_o1 = 2'b00;
                ALU_operation_o1 = 4'b0001;
            end
            else if(funct_i == 6'b010110) begin     // OR
                FURslt_o1 = 2'b00;
                ALU_operation_o1 = 4'b0000;
            end
            else if(funct_i == 6'b010101) begin     // NOR
                FURslt_o1 = 2'b00;
                ALU_operation_o1 = 4'b1101;
            end
            else if(funct_i == 6'b100000) begin     // SLT
                FURslt_o1 = 2'b00;
                ALU_operation_o1 = 4'b0111;          // !!!
            end
            else if(funct_i == 6'b000000) begin     // SLL
                FURslt_o1 = 2'b01;
                ALU_operation_o1 = 4'b0110;
            end
            else if(funct_i == 6'b000010) begin     // SRL
                FURslt_o1 = 2'b01;
                ALU_operation_o1 = 4'b0110;
            end
        end
    endcase
end




assign ALU_operation_o = ALU_operation_o1;
assign FURslt_o = FURslt_o1;
endmodule     
