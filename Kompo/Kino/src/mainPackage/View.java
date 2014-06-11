package mainPackage;

import java.awt.BorderLayout;
import java.awt.EventQueue;
import java.awt.Font;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.border.EmptyBorder;

public class View extends JFrame {
	
	private JLabel title1 = new JLabel("Witaj!");
	private JLabel title2 = new JLabel("Wybierz konto:");
	private JButton userButton = new JButton("Uzytkownik");
	private JButton adminButton = new JButton("Administrator");
	
	public UserMenu um;
	public AdminMenu am;
	
	/**
	 * Create the frame.
	 */
	public View() {
		JPanel contentPane = new JPanel();
		
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setSize(300,280);

		contentPane.add(title1);
		contentPane.add(title2);
		contentPane.add(userButton);
		contentPane.add(adminButton);
		
		contentPane.setLayout(null);
		title1.setBounds(100, 10, 100, 50);
		title1.setFont(new Font("Courier New", 2, 20));
		title2.setBounds(60, 30, 200, 50);
		title2.setFont(new Font("Courier New", 2, 20));
		userButton.setBounds(65, 100, 150, 30);
		adminButton.setBounds(65, 150, 150, 30);
		
		this.add(contentPane);
	}
	
	void addUserButtonListener(ActionListener listenForUserButton){
		userButton.addActionListener(listenForUserButton);
	}
	
	void addAdminButtonListener(ActionListener listenForAdminButton){
		adminButton.addActionListener(listenForAdminButton);
	}
	
	void createUserMenu(){
		um = new UserMenu();
	}
	
	void createAdminMenu(){
		am = new AdminMenu();
	}
}
