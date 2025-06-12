describe('Add TODO test', () => {
  it('should add successfully', () => {
    cy.visit('http://192.168.198.1:5173/')

    cy.get('#name').type("second test")
    cy.get("#description").type('please work')
    // check if i can click the button
    cy.get('button').first().should('be.enabled')
    cy.get('button').first().click()

    // get the first todo
    cy.get('.Card--text').last().should('contain', 'second test')
    cy.get('.Card--text').last().should('contain', 'please work')
  })
})
