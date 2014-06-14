package mainPackage.View;

import java.awt.Dimension;
import java.awt.Font;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

@SuppressWarnings("serial")
public class CreationWindow extends JFrame {
	private JPanel windowPane = new JPanel();
	JLabel userTitle;
	public CreationWindow(String txt){
		userTitle = new JLabel(txt);
		this.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		this.setSize(350, 200);
		
		this.add(userTitle);
		this.setLayout(null);
		
		userTitle.setBounds(30, 50, 300, 50);
		userTitle.setFont(new Font("Courier New", 1, 18));
		
		this.add(windowPane);
		this.setVisible(true);
	}
}
