module D_FF_AR(input D, Clk, Rst, output Q);
    reg Q;
    always @ (posedge Clk, negedge Rst)
        if(!Rst) Q <= 1'b0;
        else Q <= D;
endmodule