module Lab3_D_Latch_gatelevel (input D, E, output Q, Qb);
    wire D, E;
    wire Q, Qb;
    wire D_inv, and_result1, and_result2;
    not #(1) not1(D_inv, D);
    and #(1) and1(and_result1, D_inv, E);
    and #(1) and2(and_result2, E, D);
    nor #(1) nor1(Q, and_result1, Qb);
    nor #(1) nor2(Qb, and_result2, Q);
endmodule