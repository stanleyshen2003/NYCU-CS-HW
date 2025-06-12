.data
msg1:	.asciiz "Enter first number: "
msg2:   .asciiz "Enter second number: "
msg3:	.asciiz "The GCD is: "
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
 		li      $v0, 5
  		syscall 
  		move    $a1, $v0      		

# print msg2 on the console interface
		li      $v0, 4	
		la      $a0, msg2
		syscall
 
# read the input integer in $v0
 		li      $v0, 5
  		syscall
  		move    $a2, $v0
# output msh3
		li      $v0, 4
		la      $a0, msg3
		syscall
# do GCD
		jal GCD
		
		move $a0, $v0
		li $v0, 1
		syscall
		
		li $v0, 10
  		syscall
 
#------------------------- procedure GCD -----------------------------
 
.text
GCD:		
		div $a1, $a2
		mfhi $t0
		bne $t0, $zero, again
		move $v0, $a2
		jr $ra
again:
		move $a1, $a2
		move $a2, $t0
		j GCD
		
		
		