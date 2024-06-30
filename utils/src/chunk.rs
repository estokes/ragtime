use ragtime::doc::DocStore;
use anyhow::Result;

fn main() -> Result<()> {
    const DOC: &str = "/home/eric/Downloads/Fowl Engine.txt";
    let mut store = DocStore::new(128);
    for chunk in store.add_document(DOC, 256, 128)? {
        let (id, txt) = chunk?;
        println!("---------------------- CHUNK {id:?} ---------------------\n{txt}");
    }
    Ok(())
}
