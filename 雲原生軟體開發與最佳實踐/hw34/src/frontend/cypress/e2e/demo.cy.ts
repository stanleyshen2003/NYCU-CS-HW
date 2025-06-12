describe('My Todo Website Test', () => {
  it('Should load my website "/" and contain title "My Todos"', () => {
    cy.visit('/')
    cy.contains('My Todos')
  })
})
