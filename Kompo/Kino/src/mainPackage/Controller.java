package mainPackage;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;

public class Controller {
	private View theView;
	private Model theModel;
	
	public Controller(View theView, Model theModel){
		this.theView = theView;
		this.theModel = theModel;
		
		this.theView.addUserButtonListener(userButtonListener);
		this.theView.addAdminButtonListener(adminButtonListener);
	}
	
	ActionListener userButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.setVisible(false);
			theView.createUserMenu();
			theView.um.addBuyButtonListener(buyButtonListener);
			theView.um.disableButton(theView.um.getBuyButton());
			theView.um.addBookButtonListener(bookButtonListener);
			theView.um.disableButton(theView.um.getBookButton());
			theView.um.addBackButtonListener(backButtonListener);
			theView.um.getUserListSelection().addListSelectionListener(new SelectionListener());
			updateRepertoireTable();
		}
	};
	
	ActionListener adminButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.setVisible(false);
			theView.createAdminMenu();
		}
	};
	
	ActionListener buyButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			System.out.println("Kupi³eœ bilet");
		}
	};
	
	ActionListener bookButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			System.out.println("Zarezerwowa³eœ bilet");
		}
	};
	
	ActionListener backButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.setVisible(true);
			theView.um.getUserMenu().setVisible(false);
		}
	};
	
	class SelectionListener implements ListSelectionListener {
		public String txt;
		@Override
		public void valueChanged(ListSelectionEvent e) {
			// TODO Auto-generated method stub
			if(e.getValueIsAdjusting()) return;
			int row = theView.um.getUserTable().getSelectedRow();
			if(row < 0) return;
			int col = theView.um.getUserTable().getSelectedColumn();
			if(col < 0) return;
			
			theView.um.enableButton(theView.um.getBuyButton());
			theView.um.enableButton(theView.um.getBookButton());
		}
	};
	
	public void updateRepertoireTable()
	{
		SelectionController updater = new SelectionController(theModel.repertoire);
		Object[][] newContent = updater.getRepertoireAsObjects();
		theView.um.setTableContent(newContent);
	}
}
