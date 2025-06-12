module Lab2_4_bit_CLA_beh (input [3:0] A, B, input cin, output reg [3:0] S, output reg cout);

    reg [3:0] G;
    reg [3:0] P;
    reg [4:0] C;

    always @(*) begin

        G = A & B;
        P = A ^ B;
        C[0] = cin;
        C[1] = G[0] | (P[0] & cin);
        C[2] = G[1] | (P[1] & G[0]) | (P[1] & P[0] & C[0]);
        C[3] = G[2] | (P[2] & G[1]) | (P[2] & P[1] & G[0]) | (P[2] & P[1] & P[0] & C[0]);
        C[4] = G[3] | (P[3] & G[2]) | (P[3] & P[2] & G[1]) | (P[3] & P[2] & P[1] & G[0]) | (P[3] & P[2] & P[1] & P[0] & C[0]);
        
    end

    always @(*) begin
        S[0] = A[0] ^ B[0] ^ C[0];
        S[1] = A[1] ^ B[1] ^ C[1];
        S[2] = A[2] ^ B[2] ^ C[2];
        S[3] = A[3] ^ B[3] ^ C[3];
        cout = C[4];
    end

endmodule
