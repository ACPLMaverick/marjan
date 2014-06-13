package mainPackage.View;

import java.awt.Font;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

public class SmallWindow extends JFrame {
	private JFrame window = new JFrame();
	private JPanel windowPane = new JPanel();
	
	public SmallWindow(String txt){
		JLabel userTitle = new JLabel(txt);
		this.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		this.setSize(350, 200);
		
		this.add(userTitle);
		this.setLayout(null);
		
		userTitle.setBounds(30, 50, 300, 50);
		userTitle.setFont(new Font("Courier New", 2, 18));
		
		this.add(windowPane);
		this.setVisible(true);
	}
}
