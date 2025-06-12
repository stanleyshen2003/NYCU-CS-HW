module	t_Lab2_BCD_9c();
	wire[3:0]	BCD_9c1, BCD_9c2;
	reg[3:0]	BCD;
	
	Lab2_BCD_9c_df	BCD_9c_df(BCD, BCD_9c1);
    Lab2_BCD_9c_beh BCD_9c_beh(BCD, BCD_9c2);
    integer i;
    
	initial	begin
		for(i=0;i<16;i++) begin
            BCD[0] = i[0];
            BCD[1] = i[1];
            BCD[2] = i[2];
            BCD[3] = i[3];
            #15;
        end

	end
	initial #200 $finish;

	//dump the result of simulation
	initial begin
		$dumpfile("Lab2_BCD_9c.vcd");
		$dumpvars;
	end
endmodule