module Lab2_4_bit_CLA_gate (input [3:0] A, B, input cin, output [3:0] S, output cout);

    wire [3:0] G;
    wire [3:0] P;
    wire [2:0] C;

    and #(2) and1(G[0], A[0], B[0]);
    and #(2) and2(G[1], A[1], B[1]);
    and #(2) and3(G[2], A[2], B[2]);
    and #(2) and4(G[3], A[3], B[3]);

    xor #(4) xor1(P[0], A[0], B[0]);
    xor #(4) xor2(P[1], A[1], B[1]);
    xor #(4) xor3(P[2], A[2], B[2]);
    xor #(4) xor4(P[3], A[3], B[3]);

    wire p0cin, p1g0, p1p0cin, p2g1, p2p1g0, p2p1p0cin, p3g2, p3p2g1, p3p2p1g0, p3p2p1p0cin;

    and #(2) and5(p0cin, P[0], cin);
    or #(2) or1(C[0], G[0], p0cin);

    and #(2) and6(p1g0, P[1], G[0]);
    and #(2) and7(p1p0cin, P[0], cin, P[1]);
    or #(2) or2(C[1], G[1], p1p0cin, p1g0);

    and #(2) and8(p2g1, P[2], G[1]);
    and #(2) and9(p2p1g0, P[2], G[0], P[1]);
    and #(2) and10(p2p1p0cin, P[2], P[0], P[1], cin);
    or #(2) or3(C[2], G[2], p2g1, p2p1g0, p2p1p0cin);

    and #(2) and11(p3g2, P[3], G[2]);
    and #(2) and12(p3p2g1, P[3], G[1], P[2]);
    and #(2) and13(p3p2p1g0, P[3], P[2], P[1], G[0]);
    and #(2) and14(p3p2p1p0cin, P[3], P[2], P[0], P[1], cin);
    or #(2) or4(cout, G[3], p3g2, p3p2g1, p3p2p1g0, p3p2p1p0cin);
    
    xor #(4) xor5(S[0], P[0], cin);
    xor #(4) xor6(S[1], P[1], C[0]);   
    xor #(4) xor7(S[2], P[2], C[1]); 
    xor #(4) xor8(S[3], P[3], C[2]);

endmodule
