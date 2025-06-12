module Lab2_4_bit_CLA_df (input [3:0] A, B, input cin, output [3:0] S, output cout);

    wire [3:0] G;
    wire [3:0] P;
    wire [4:0] C;

    assign G = A & B;
    assign P = A ^ B;
    assign C[0] = cin;
    assign C[1] = G[0] | (P[0] & cin);
    assign C[2] = G[1] | (P[1] & G[0]) | (P[1] & P[0] & C[0]);
    assign C[3] = G[2] | (P[2] & G[1]) | (P[2] & P[1] & G[0]) | (P[2] & P[1] & P[0] & C[0]);
    assign C[4] = G[3] | (P[3] & G[2]) | (P[3] & P[2] & G[1]) | (P[3] & P[2] & P[1] & G[0]) | (P[3] & P[2] & P[1] & P[0] & C[0]);
    

    assign S[0] = A[0] ^ B[0] ^ C[0];
    assign S[1] = A[1] ^ B[1] ^ C[1];
    assign S[2] = A[2] ^ B[2] ^ C[2];
    assign S[3] = A[3] ^ B[3] ^ C[3];
    assign cout = C[4];

endmodule
