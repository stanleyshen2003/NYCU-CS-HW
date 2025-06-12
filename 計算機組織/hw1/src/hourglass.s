.data
msg1:	.asciiz "Enter the number n = "
msg2:   .asciiz " "
msg3:	.asciiz "*"
msg4:   .asciiz "\n"

.text
.globl main
#------------------------- main -----------------------------
main:
# print msg1 on the console interface
		li      $v0, 4
		la      $a0, msg1
		syscall
 
# read the input integer in $v0
 		li      $v0, 5          	# call system call: read integer
  		syscall                 	# run the syscall
  		move    $t0, $v0      		# store input in $a0 (set arugument of function)

# process data
		addi $t1, $t0, 1 			
		srl $t1,$t1, 1			# t1 = temp
		addi $t2, $zero, 1		# t2 = 1 (mask)
		and $t2, $t2, $t0
		add $t3, $zero, $zero		# t3 = i, set i for loop
		beq $t2, $zero, drawup
		addi $t1, $t1, -1		# temp --
# start drawing up part
drawup:
		add $t4, $zero, $zero		# t4 = j = 0
		slt $t5, $t3, $t1		# t5 for exit first loop
		beq $zero, $t5, setdrawdown
		
# draw up empty spaces
upEmpty:
		slt $t5, $t3, $t4		# i<j
		bne $zero, $t5, resetj
		
		li $v0, 4			# call system call: print string
		la $a0, msg2			# load address of string into $a0
		syscall
		
		addi $t4, $t4, 1
		j upEmpty
resetj:		add $t4, $zero, $zero
		add $t6, $t3, $t3
		sub $t6, $t0, $t6
		
# draw up stars
upstar:
		slt $t5, $t4, $t6		# t5 for exit first loop
		beq $zero, $t5, endtop
		
		li $v0, 4			# call system call: print string
		la $a0, msg3			# load address of string into $a0
		syscall

		addi $t4, $t4, 1
		j upstar
		
# finish iteration of 1st loop
endtop:		
		addi $t3, $t3, 1		# i++
		li $v0, 4			# call system call: print string
		la $a0, msg4			# load address of string into $a0
		syscall
  		j drawup
  		
setdrawdown:
		addi $t2, $t0, 1
		srl $t2, $t2, 1
		addi $t3, $t2, -1
		
drawDown:
		slt  $t5, $t3, $zero
		bne $t5, $zero, finish 
		add $t4, $zero, $zero
downEmpty:
		slt $t5, $t3, $t4
		bne $t5, $zero, setDownStar
		
		li $v0, 4			# call system call: print string
		la $a0, msg2			# load address of string into $a0
		syscall
		
		addi $t4, $t4, 1
		j downEmpty
setDownStar:
		add $t4, $zero, $zero
		sll $t6, $t3, 1
		sub $t6, $t0, $t6
downStar:
		slt $t5, $t4, $t6
		beq $zero, $t5, endDown
		
		li $v0, 4			# call system call: print string
		la $a0, msg3			# load address of string into $a0
		syscall
			
		addi $t4, $t4, 1
		j downStar	
endDown:
		addi $t3, $t3, -1
		li $v0, 4			# call system call: print string
		la $a0, msg4			# load address of string into $a0
		syscall
  		j drawDown
		
finish:	
		li $v0, 10			# call system call: exit
  		syscall				# run the syscall
