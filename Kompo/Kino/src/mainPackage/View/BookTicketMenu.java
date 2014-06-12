package mainPackage.View;

import javax.swing.JFrame;

public class BookTicketMenu extends TicketMenu {
	
	public BookTicketMenu(){
		this.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		this.setSize(300,280);
		
		this.setLayout(null);
		
		this.add(getTicketMenuPane());
		this.setVisible(true);
	}
}
