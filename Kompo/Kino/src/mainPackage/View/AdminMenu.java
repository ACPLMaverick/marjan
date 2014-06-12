package mainPackage.View;

import java.awt.Font;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

public class AdminMenu {
	JFrame adminMenu = new JFrame();
	JPanel adminPane = new JPanel();
	JLabel adminTitle = new JLabel("Witaj w koncie administratora!");
	
	public AdminMenu(){
		adminMenu.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		adminMenu.setSize(380,280);
		
		adminPane.add(adminTitle);
		
		adminPane.setLayout(null);
		adminTitle.setBounds(15, 10, 350, 50);
		adminTitle.setFont(new Font("Courier New", 2, 18));
		
		adminMenu.add(adminPane);
		adminMenu.setVisible(true);
	}
}
