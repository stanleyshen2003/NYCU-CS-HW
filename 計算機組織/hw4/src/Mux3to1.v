module Mux3to1( data0_i, data1_i, data2_i, select_i, data_o );

parameter size = 0;			   
			
//I/O ports               
input wire	[size-1:0] data0_i;          
input wire	[size-1:0] data1_i;
input wire	[size-1:0] data2_i;
input wire	[2-1:0] select_i;
output wire	[size-1:0] data_o; 

//Main function

reg [size-1:0] data_o1;
always@(data0_i or data1_i or data2_i or select_i)
begin
  if(select_i == 2'b00)
	data_o1 = data0_i;
  else if(select_i == 2'b01)
	data_o1 = data1_i;
  else if(select_i == 2'b10)
	data_o1 = data2_i;
end

assign data_o = data_o1;

endmodule      
