package mainPackage.View;

import java.awt.Dimension;
import java.awt.Font;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

// TODO: Auto-generated Javadoc
/**
 * Klasa reprezentuje proste okno z samym tekstem.
 */
@SuppressWarnings("serial")
public class SmallWindow extends JFrame {
	
	private JPanel windowPane = new JPanel();
	
	/**
	 * Tworzy proste okienko z pojedynczym tekstem.
	 *
	 * @param txt teksty wyswietlany w oknie.
	 */
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
	
	/**
	 * Tworzy proste okienko z dwoma tekstami.
	 *
	 * @param txtBig tekst napisany wiekszym rozmiarem czcionki.
	 * @param txtSmall tekst napisany mniejszym rozmiarem czcionki.
	 */
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
	
	/**
	 * Tworzy proste okienko z tytulem i dwoma tekstami.
	 *
	 * @param txt tytul.
	 * @param name1 tekst napisany mniejszym rozmiarem czcionki.
	 * @param name2 tekst napisany mniejszym rozmiarem czcionki.
	 */
	public SmallWindow(String txt, String name1, String name2){
		JLabel userTitle = new JLabel(txt);
		JLabel userName1 = new JLabel(name1);
		JLabel userName2 = new JLabel(name2);
		this.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		this.setSize(350, 200);
		
		this.add(userTitle);
		this.add(userName1);
		this.add(userName2);
		this.setLayout(null);
		
		userTitle.setBounds(30, 20, 300, 50);
		userTitle.setFont(new Font("Courier New", 1, 18));
		
		userName1.setBounds(30, 50, 300, 100);
		userName1.setFont(new Font("Courier New", 0, 16));
		
		userName2.setBounds(30, 70, 300, 100);
		userName2.setFont(new Font("Courier New", 0, 16));
		
		this.add(windowPane);
		this.setVisible(true);
	}
}
