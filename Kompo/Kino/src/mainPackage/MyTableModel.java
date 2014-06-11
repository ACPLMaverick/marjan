package mainPackage;

import javax.swing.table.AbstractTableModel;

public class MyTableModel extends AbstractTableModel{
	private String[] columnNames = {"Tytu³", "Godzina", "Dzieñ", "Sala", "Cena"};
	private Object[][] test = {{"Ojciec Chrzestny", "17:00", "10-06-2014", "4", "14.00"}, 
							   {"Taksówkarz", "15:00", "12-06-2014", "5", "15.00"}
	};
	
	public int getColumnCount() {
		return columnNames.length;
	}

	public int getRowCount() {
		return test.length;
	}
	
	public String getColumnName(int col){
		return columnNames[col];
	}

	public Object getValueAt(int row, int col) {
		return test[row][col];
	}
	
	@SuppressWarnings("unchecked")
	public Class getColumnClass(int c){
		return getValueAt(0, c).getClass();
	}    
	
	public boolean isCellEditable(int row, int col) {
        if (col < 6) {
            return false;
        } else {
            return true;
        }
    }
}
