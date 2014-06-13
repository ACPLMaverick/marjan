package mainPackage.View;

import java.awt.Font;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

@SuppressWarnings("serial")
public class SmallWindow extends JFrame {
	private JPanel windowPane = new JPanel();
	
	public SmallWindow(String txt){
		JLabel userTitle = new JLabel(txt);
		this.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		this.setSize(350, 200);
		
		this.add(userTitle);
		this.setLayout(null);
		
		userTitle.setBounds(30, 50, 300, 50);
		userTitle.setFont(new Font("Courier New", 1, 18));
		
		this.add(windowPane);
		this.setVisible(true);
	}
	
	public SmallWindow(String txtBig, String txtSmall){
		JLabel userTitle = new JLabel(txtBig);
		JLabel description = new JLabel(txtSmall);
		this.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		this.setSize(350, 200);
		
		this.add(userTitle);
		this.add(description);
		this.setLayout(null);
		
		userTitle.setBounds(30, 20, 300, 50);
		userTitle.setFont(new Font("Courier New", 1, 18));
		
		description.setBounds(30, 80, 300, 100);
		description.setFont(new Font("Courier New", 0, 10));
		
		this.add(windowPane);
		this.setVisible(true);
	}
}
