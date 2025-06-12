module Lab3_D_FF_gatelevel(input D, Clk, output Q, Qb);
    wire one2to, notclk, doublenot, nouse;
    not #(1) not1(notclk, Clk);
    not #(1) not2(doublenot, notclk);
    Lab3_D_Latch_gatelevel FF1(D, notclk, one2to, nouse);
    Lab3_D_Latch_gatelevel FF2(one2to, doublenot, Q, Qb);
endmodule