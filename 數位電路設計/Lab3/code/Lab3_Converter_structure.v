module Lab3_Converter_structure(input X, Clk, Rst, output Z);

    wire ffinput1, ffinput2, ffinput3;
    wire ffoutput1, ffoutput2, ffoutput3;

    assign ffinput1 = (~X & ~ffoutput2) | (ffoutput1 & ~ffoutput2 & ~ffoutput3);
    assign ffinput2 = (ffoutput1 & ffoutput3) | (X & ~ffoutput2 & ffoutput3) | ( ~ffoutput1 & ffoutput2 & ~ffoutput3);
    assign ffinput3 = (~ffoutput3 & (ffoutput1 ^ ffoutput2)) | (X & ~ffoutput2 & (ffoutput1 | ~ffoutput3)) | (~X & ~ffoutput1 & ~ffoutput2 & ffoutput3);
    assign Z = (~X & ~ffoutput2 & (~ffoutput1 | ffoutput3)) | (X & ~ffoutput1 & ffoutput2) | (X & ffoutput1 & ~ffoutput2 & ~ffoutput3);

    D_FF_AR first(ffinput1, Clk, Rst, ffoutput1);
    D_FF_AR second(ffinput2, Clk, Rst, ffoutput2);
    D_FF_AR three(ffinput3, Clk, Rst, ffoutput3);
endmodule