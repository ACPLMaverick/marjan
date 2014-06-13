package mainPackage.View;

import javax.swing.table.AbstractTableModel;

//przyda�oby si� i trzeba zrobi� te� tabel� przychod�w i wydatk�w. Przyj�te kolumny: TYP, DATA, ILO�� HAJSU
//hajs si� musi zgadza�
//przyda�oby si� zrobi� te� tabel� aktualnie kupionych bilet�w do wy�wietlenia gdzie� obok. Przyj�te kolumny: TYTU� FILMU, DATA, CENA

public class MyTableModel extends AbstractTableModel{
	private String[] columnNames = {"Tytu�", "Godzina", "Dzie�", "Zaj�te miejsca", "Cena"};
	private Object[][] content = {{"Ojciec Chrzestny", "17:00", "10-06-2014", "4", "14.00"}, 
							   	 {"Taks�wkarz", "15:00", "12-06-2014", "5", "15.00"}
	};
	private String type;
	
	public MyTableModel(){};
	
	public MyTableModel(String[] columnNames, Object[][] content, String type){
		this.columnNames = columnNames;
		setContent(content);
		this.type = type;
	}
	
	public int getColumnCount() {
		return columnNames.length;
	}

	public int getRowCount() {
		return content.length;
	}
	
	public String getColumnName(int col){
		return columnNames[col];
	}

	public Object getValueAt(int row, int col) {
		return content[row][col];
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
	
	public void setContent(Object[][] newContent) { this.content = newContent; }
}
