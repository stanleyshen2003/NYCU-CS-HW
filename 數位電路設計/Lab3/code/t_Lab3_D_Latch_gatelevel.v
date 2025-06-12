module t_Lab3_D_Latch_gatelevel();

reg D, E;
wire Q, Qb ;
Lab3_D_Latch_gatelevel D_latch(D, E, Q, Qb);

    initial fork
        #0 D = 0; 
        #0 E = 0;
        #5 E = 1;
        #10 E = 0;
        #15 D = 1;
        #20 E = 1;
        #25 D = 0; 
        #25 E = 0;
        #30 E = 1;
        #35 E = 0;
        #50 D = 1; 
        #50 E = 1;
        #51 E = 0;
        #55 E = 1;   
        #60 $finish;
    join
    initial begin
		$dumpfile("t_Lab3_D_Latch_gatelevel.vcd");
		$dumpvars;
	end



endmodule