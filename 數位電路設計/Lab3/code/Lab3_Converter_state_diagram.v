module Lab3_Converter_state_diagram (input X, Clk, Rst, output Z);

    reg [2:0] state, nextstate;
    reg Z;
    parameter S0 = 3'b000, S1 = 3'b001, S2 = 3'b010, S3 = 3'b011, S4 = 3'b100, S5 = 3'b101, S6 = 3'b110;

    always @(posedge Clk, negedge Rst)  // handle the update
        if(Rst == 0) state <= S0;
        else state <= nextstate;

    always @(state, X)              // handle the change of states
        case(state)
            S0: if(~X) nextstate = S4; else nextstate = S1;
            S1: if(~X) nextstate = S5; else nextstate = S2;
            S2: nextstate = S3;
            S3: nextstate = S0;
            S4: nextstate = S5;
            S5: if(~X) nextstate = S6; 
                else nextstate = S3;
            S6: nextstate = S0;
        endcase

    always @ (state, X)             // handle the output
        case (state)
            S0, S1, S5, S6: Z = ~X;
            S2, S3, S4: Z = X; 
        endcase

endmodule