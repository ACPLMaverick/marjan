package mainPackage.View;

import javax.swing.JFrame;
import javax.swing.JPanel;

// TODO: Auto-generated Javadoc
/**
 * The Class TicketMenu.
 */
public class TicketMenu extends JFrame {
	
	private JFrame ticketMenuFrame = new JFrame();
	private JPanel ticketMenuPane = new JPanel();
	
	/**
	 * Zwraca ramke JFrame menu biletow.
	 *
	 * @return Ramke JFrame menu biletow.
	 */
	public JFrame getTicketMenuFrame(){
		return ticketMenuFrame;
	}
	
	/**
	 * Zwraca panel JPanel menu biletow.
	 *
	 * @return Panel JPanel menu biletow.
	 */
	public JPanel getTicketMenuPane(){
		return ticketMenuPane;
	}
}
