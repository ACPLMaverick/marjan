package mainPackage.View;

import javax.swing.JFrame;
import javax.swing.JPanel;

public class TicketMenu extends JFrame {
	private JFrame ticketMenuFrame = new JFrame();
	private JPanel ticketMenuPane = new JPanel();
	
	public JFrame getTicketMenuFrame(){
		return ticketMenuFrame;
	}
	public JPanel getTicketMenuPane(){
		return ticketMenuPane;
	}
}
