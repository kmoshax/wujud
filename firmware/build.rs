fn main() {
    #[cfg(target_os = "espidf")]
    embuild::espidf::sysenv_output();
}
